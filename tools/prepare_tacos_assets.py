#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_number(value):
    number = float(value)
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def infer_split(input_path: Path, explicit_split: str | None) -> str:
    if explicit_split:
        return explicit_split
    stem = input_path.stem.lower()
    if stem.startswith("train"):
        return "train"
    if stem.startswith("val"):
        return "val"
    if stem.startswith("test"):
        return "test"
    return "unknown"


def build_manifest(
    payload: dict,
    split: str,
    video_root: Path | None,
    markit_position: str,
    markit_font_size: int,
    markit_color: str,
) -> list[dict]:
    manifest = []
    for video_name, sample in payload.items():
        if not isinstance(sample, dict):
            raise ValueError(f"{video_name}: expected dict payload")

        sentences = sample.get("sentences")
        timestamps = sample.get("timestamps")
        fps = sample.get("fps")
        num_frames = sample.get("num_frames")
        duration = sample.get("duration", sample.get("video_duration", num_frames))

        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            raise ValueError(f"{video_name}: missing list fields 'sentences'/'timestamps'")
        if len(sentences) != len(timestamps):
            raise ValueError(
                f"{video_name}: sentence count {len(sentences)} does not match timestamp count {len(timestamps)}"
            )
        if fps is None:
            raise ValueError(f"{video_name}: missing 'fps'")
        if num_frames is None:
            raise ValueError(f"{video_name}: missing 'num_frames'")

        video_path = video_root / video_name if video_root is not None else None
        video_exists = video_path.exists() if video_path is not None else None

        for idx, (query, timestamp) in enumerate(zip(sentences, timestamps), start=1):
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"{video_name}: invalid timestamp at index {idx}: {timestamp}")

            start_time, end_time = timestamp
            manifest.append(
                {
                    "id": f"{Path(video_name).stem}_{idx}",
                    "video": video_name,
                    "video_stem": Path(video_name).stem,
                    "query": normalize_text(query),
                    "start_time": normalize_number(start_time),
                    "end_time": normalize_number(end_time),
                    "fps": normalize_number(fps),
                    "num_frames": int(num_frames),
                    "duration": int(duration),
                    "video_duration": int(duration),
                    "split": split,
                    "time_unit": "frame",
                    "task": "moment_retrieval",
                    "markit_position": markit_position,
                    "markit_font_size": int(markit_font_size),
                    "markit_color": markit_color,
                    "video_path": str(video_path) if video_path is not None else None,
                    "video_exists": video_exists,
                }
            )

    return manifest


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_json")
    parser.add_argument("--split")
    parser.add_argument("--video_root")
    parser.add_argument("--markit_position", default="bottom_right")
    parser.add_argument("--markit_font_size", type=int, default=38)
    parser.add_argument("--markit_color", default="red")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    summary_path = Path(args.summary_json) if args.summary_json else None
    video_root = Path(args.video_root) if args.video_root else None

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected grouped TACoS payload with top-level dict")

    split = infer_split(input_path, args.split)
    manifest = build_manifest(
        payload=payload,
        split=split,
        video_root=video_root,
        markit_position=args.markit_position,
        markit_font_size=args.markit_font_size,
        markit_color=args.markit_color,
    )
    write_json(manifest, output_path)

    unique_videos = len({item["video"] for item in manifest})
    existing_videos = len({item["video"] for item in manifest if item["video_exists"] is True})
    summary = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "split": split,
        "time_unit": "frame",
        "num_records": len(manifest),
        "num_videos": unique_videos,
        "video_root": str(video_root) if video_root is not None else None,
        "videos_found_on_disk": existing_videos,
        "markit_defaults": {
            "position": args.markit_position,
            "font_size": args.markit_font_size,
            "color": args.markit_color,
        },
    }
    if summary_path is not None:
        write_json(summary, summary_path)

    print(
        f"wrote {output_path} with {len(manifest)} records "
        f"across {unique_videos} videos (split={split})"
    )
    if summary_path is not None:
        print(f"wrote summary {summary_path}")


if __name__ == "__main__":
    main()
