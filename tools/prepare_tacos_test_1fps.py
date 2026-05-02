#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2


def load_grouped_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected grouped TACoS payload with top-level dict")
    return payload


def probe_video(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or num_frames <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {video_path}: fps={fps}, num_frames={num_frames}"
        )
    return fps, num_frames, width, height


def run_ffmpeg_extract_1fps(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "fps=1",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_timestamp_pair(timestamp, raw_fps: float, output_num_frames: int) -> list[int]:
    if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
        raise ValueError(f"Invalid timestamp: {timestamp}")

    start_raw = float(timestamp[0])
    end_raw = float(timestamp[1])

    start_1fps = math.floor(start_raw / raw_fps)
    end_1fps = math.ceil(end_raw / raw_fps)

    start_1fps = max(0, start_1fps)
    end_1fps = max(start_1fps, end_1fps)

    if output_num_frames > 0:
        max_idx = output_num_frames - 1
        start_1fps = min(start_1fps, max_idx)
        end_1fps = min(end_1fps, max_idx)

    return [int(start_1fps), int(end_1fps)]


def convert_video_entry(video_name: str, sample: dict, raw_fps: float, output_num_frames: int) -> dict:
    sentences = sample.get("sentences")
    timestamps = sample.get("timestamps")
    if not isinstance(sentences, list) or not isinstance(timestamps, list):
        raise ValueError(f"{video_name}: missing list fields 'sentences'/'timestamps'")
    if len(sentences) != len(timestamps):
        raise ValueError(
            f"{video_name}: sentence count {len(sentences)} does not match timestamp count {len(timestamps)}"
        )

    new_name = f"{Path(video_name).stem}.mp4"
    converted_timestamps = [
        convert_timestamp_pair(ts, raw_fps=raw_fps, output_num_frames=output_num_frames)
        for ts in timestamps
    ]

    return new_name, {
        "timestamps": converted_timestamps,
        "sentences": sentences,
        "fps": 1.0,
        "num_frames": output_num_frames,
        "duration": output_num_frames,
        "video_duration": output_num_frames,
        "source_video": video_name,
        "source_fps": raw_fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare TACoS test split as 1fps mp4 videos.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--raw_video_root", required=True)
    parser.add_argument("--output_video_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_manifest", required=False)
    parser.add_argument("--output_summary", required=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    raw_video_root = Path(args.raw_video_root)
    output_video_root = Path(args.output_video_root)
    output_json = Path(args.output_json)
    output_manifest = Path(args.output_manifest) if args.output_manifest else None
    output_summary = Path(args.output_summary) if args.output_summary else None

    payload = load_grouped_json(input_json)
    output_video_root.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_summary:
        output_summary.parent.mkdir(parents=True, exist_ok=True)

    converted = {}
    manifest = []

    for video_name, sample in payload.items():
        src_path = raw_video_root / video_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source video: {src_path}")

        raw_fps, raw_num_frames, width, height = probe_video(src_path)
        dst_name = f"{Path(video_name).stem}.mp4"
        dst_path = output_video_root / dst_name

        if args.overwrite or not dst_path.exists():
            run_ffmpeg_extract_1fps(src_path, dst_path)

        out_fps, out_num_frames, out_w, out_h = probe_video(dst_path)
        if abs(out_fps - 1.0) > 1e-4:
            raise RuntimeError(f"{dst_path}: expected 1fps output, got {out_fps}")

        new_name, new_sample = convert_video_entry(
            video_name=video_name,
            sample=sample,
            raw_fps=raw_fps,
            output_num_frames=out_num_frames,
        )
        converted[new_name] = new_sample

        manifest.append(
            {
                "source_video": video_name,
                "source_path": str(src_path),
                "source_fps": raw_fps,
                "source_num_frames": raw_num_frames,
                "source_size": [width, height],
                "output_video": new_name,
                "output_path": str(dst_path),
                "output_fps": out_fps,
                "output_num_frames": out_num_frames,
                "output_size": [out_w, out_h],
                "num_queries": len(sample["sentences"]),
            }
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    if output_manifest:
        with output_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    if output_summary:
        summary = {
            "input_json": str(input_json),
            "raw_video_root": str(raw_video_root),
            "output_video_root": str(output_video_root),
            "output_json": str(output_json),
            "num_videos": len(converted),
            "num_queries": sum(len(v["sentences"]) for v in converted.values()),
            "output_fps": 1.0,
        }
        with output_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(converted)} videos to {output_video_root}")
    print(f"wrote annotations to {output_json}")


if __name__ == "__main__":
    main()
