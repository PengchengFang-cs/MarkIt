import argparse
import json
import math
import os
import re

import cv2
import numpy as np


def cal_iou(seg_a, seg_b):
    max0 = max(seg_a[0], seg_b[0])
    min0 = min(seg_a[0], seg_b[0])
    max1 = max(seg_a[1], seg_b[1])
    min1 = min(seg_a[1], seg_b[1])
    intersection = max(min1 - max0, 0)
    union = max1 - min0
    return intersection / union if union > 0 else 0.0


def parse_prediction(response_text: str):
    text = str(response_text)
    frame_match = re.search(
        r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*to\s*(?:frame\s*)?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if frame_match:
        start_value = float(frame_match.group(1))
        end_value = float(frame_match.group(2))
        return {
            "kind": "frame",
            "start": max(0, int(math.floor(start_value))),
            "end": max(0, int(math.ceil(end_value))),
        }

    second_match = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds?",
        text,
        re.IGNORECASE,
    )
    if second_match:
        start_value = float(second_match.group(1))
        end_value = float(second_match.group(2))
        return {
            "kind": "seconds",
            "start": max(0.0, start_value),
            "end": max(start_value, end_value),
        }

    return None


def probe_video_metadata(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Error opening video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if fps <= 0:
        fps = 1.0
    return float(fps), max(total_frames, 0)


def resolve_video_path(item: dict, overlay_root: str, raw_video_root: str) -> str:
    render_mode = item.get("render_mode", "")
    if render_mode == "overlay_direct":
        if not overlay_root:
            raise FileNotFoundError("overlay_root is required to score overlay_direct results without metadata.")
        overlay_path = os.path.join(overlay_root, f"{item['id']}_overlay.mp4")
        if not os.path.exists(overlay_path):
            raise FileNotFoundError(f"Missing overlay video: {overlay_path}")
        return overlay_path

    if not raw_video_root:
        raise FileNotFoundError("raw_video_root is required to score raw_direct results without metadata.")
    raw_video_path = os.path.join(raw_video_root, item["video"])
    if not os.path.exists(raw_video_path):
        raise FileNotFoundError(f"Missing raw video: {raw_video_path}")
    return raw_video_path


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return int(np.ceil(number / factor) * factor)


def floor_by_factor(number: float, factor: int) -> int:
    return int(np.floor(number / factor) * factor)


def infer_sampling_metadata(item: dict, args: argparse.Namespace) -> dict:
    if "source_fps" in item and "source_total_frames" in item and "effective_num_frames" in item:
        return {
            "backend": item["backend"],
            "source_fps": float(item["source_fps"]),
            "source_total_frames": int(item["source_total_frames"]),
            "effective_num_frames": int(item["effective_num_frames"]),
            "frame_label_base": int(item.get("frame_label_base", 1)),
            "sampling_strategy": item.get("sampling_strategy", ""),
            "sampling_fps_arg": item.get("sampling_fps_arg"),
            "num_frames_arg": item.get("num_frames_arg"),
        }

    video_path = resolve_video_path(item, args.overlay_root, args.raw_video_root)
    source_fps, source_total_frames = probe_video_metadata(video_path)
    backend = item["backend"]
    if backend == "internvl3":
        effective_num_frames = min(source_total_frames, args.num_sampled_frames)
        sampling_strategy = "internvl3_uniform_max_frames"
        sampling_fps_arg = None
        num_frames_arg = args.num_sampled_frames
    elif backend == "videollama3":
        sampling_fps_arg = float(source_fps)
        num_frames_arg = args.num_sampled_frames
        if sampling_fps_arg > 0 and (source_total_frames / sampling_fps_arg) < num_frames_arg:
            effective_num_frames = source_total_frames
            sampling_strategy = "videollama3_fps_sampling"
        else:
            effective_num_frames = min(source_total_frames, num_frames_arg)
            sampling_strategy = "videollama3_uniform_max_frames"
    elif backend == "qwen2_5_vl":
        if source_total_frames <= 2:
            effective_num_frames = source_total_frames
        else:
            frame_factor = 2
            fps_min_frames = 4
            fps_max_frames = 768
            sampling_fps_arg = float(source_fps)
            min_frames = ceil_by_factor(fps_min_frames, frame_factor)
            max_frames = floor_by_factor(min(fps_max_frames, source_total_frames), frame_factor)
            nframes = source_total_frames / source_fps * sampling_fps_arg
            nframes = min(max(nframes, min_frames), max_frames)
            effective_num_frames = round_by_factor(nframes, frame_factor)
        sampling_strategy = "qwen2_5_vl_smart_nframes_fps"
        sampling_fps_arg = float(source_fps)
        num_frames_arg = None
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return {
        "backend": backend,
        "source_fps": source_fps,
        "source_total_frames": source_total_frames,
        "effective_num_frames": int(effective_num_frames),
        "frame_label_base": 1,
        "sampling_strategy": sampling_strategy,
        "sampling_fps_arg": sampling_fps_arg,
        "num_frames_arg": num_frames_arg,
    }


def reconstruct_sampled_frame_indices(meta: dict) -> list[int]:
    total_frames = int(meta["source_total_frames"])
    source_fps = float(meta["source_fps"])
    backend = meta["backend"]

    if total_frames <= 0:
        return []

    if backend == "internvl3":
        effective_num_frames = int(meta["effective_num_frames"])
        if total_frames <= effective_num_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num=effective_num_frames, dtype=int).tolist()

    if backend == "videollama3":
        num_frames_arg = int(meta["num_frames_arg"])
        sampling_fps_arg = float(meta["sampling_fps_arg"])
        if sampling_fps_arg > 0 and (total_frames / source_fps) < num_frames_arg:
            segment_len = min(int(source_fps // sampling_fps_arg), total_frames)
            segment_len = max(segment_len, 1)
            return np.arange(segment_len // 2, total_frames, segment_len, dtype=int).tolist()
        if total_frames <= num_frames_arg:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num=num_frames_arg, dtype=int).tolist()

    if backend == "qwen2_5_vl":
        effective_num_frames = int(meta["effective_num_frames"])
        if total_frames <= effective_num_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num=effective_num_frames).round().astype(int).tolist()

    raise ValueError(f"Unsupported backend: {backend}")


def sampled_timestamps_from_meta(meta: dict) -> list[float]:
    sampled_indices = reconstruct_sampled_frame_indices(meta)
    source_fps = float(meta["source_fps"])
    if source_fps <= 0:
        source_fps = 1.0
    return [idx / source_fps for idx in sampled_indices]


def map_frame_prediction_to_seconds(item: dict, args: argparse.Namespace) -> tuple[float, float]:
    parsed = parse_prediction(item.get("response", ""))
    if parsed is not None and parsed["kind"] == "seconds":
        return float(parsed["start"]), float(parsed["end"])

    if parsed is not None and parsed["kind"] == "frame":
        pred_start = int(parsed["start"])
        pred_end = int(parsed["end"])
    else:
        pred_start = max(0, int(item.get("pred_start", 0)))
        pred_end = max(pred_start, int(item.get("pred_end", pred_start)))

    meta = infer_sampling_metadata(item, args)
    timestamps = sampled_timestamps_from_meta(meta)
    if not timestamps:
        return 0.0, 0.0

    base = int(meta.get("frame_label_base", 1))
    start_idx = max(0, min(len(timestamps) - 1, pred_start - base))
    end_idx = max(start_idx, min(len(timestamps) - 1, pred_end - base))
    return float(timestamps[start_idx]), float(timestamps[end_idx])


def calculate_metrics(data: list[dict]) -> tuple[float, float, float, float]:
    num_samples = len(data)
    total_iou = sum(sample["iou"] for sample in data)
    miou = (total_iou / num_samples) * 100 if num_samples else 0.0
    r_03 = sum(1 for sample in data if sample["iou"] >= 0.3) / num_samples * 100 if num_samples else 0.0
    r_05 = sum(1 for sample in data if sample["iou"] >= 0.5) / num_samples * 100 if num_samples else 0.0
    r_07 = sum(1 for sample in data if sample["iou"] >= 0.7) / num_samples * 100 if num_samples else 0.0
    return round(miou, 2), round(r_03, 2), round(r_05, 2), round(r_07, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--overlay_root", default="")
    parser.add_argument("--raw_video_root", default="")
    parser.add_argument("--num_sampled_frames", type=int, default=32)
    parser.add_argument("--no_write_back", action="store_true")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        pred_start_sec, pred_end_sec = map_frame_prediction_to_seconds(item, args)
        item["pred_start_seconds"] = pred_start_sec
        item["pred_end_seconds"] = pred_end_sec
        item["iou"] = cal_iou([pred_start_sec, pred_end_sec], [item["gt_start"], item["gt_end"]])

    if not args.no_write_back:
        with open(args.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    miou, r_03, r_05, r_07 = calculate_metrics(data)
    print(f"R@0.3: {r_03:.2f}")
    print(f"R@0.5: {r_05:.2f}")
    print(f"R@0.7: {r_07:.2f}")
    print(f"mIoU: {miou:.2f}")


if __name__ == "__main__":
    main()
