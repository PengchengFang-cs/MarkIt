#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def convert_payload(payload: dict) -> dict:
    out = {}
    for video_name, sample in payload.items():
        if not isinstance(sample, dict):
            raise ValueError(f"{video_name}: expected dict payload")
        sentences = sample.get("sentences")
        timestamps = sample.get("timestamps")
        fps = sample.get("fps")
        num_frames = sample.get("num_frames")
        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            raise ValueError(f"{video_name}: missing list fields 'sentences'/'timestamps'")
        if len(sentences) != len(timestamps):
            raise ValueError(
                f"{video_name}: sentence count {len(sentences)} does not match timestamp count {len(timestamps)}"
            )
        if num_frames is None:
            raise ValueError(f"{video_name}: missing 'num_frames'")
        duration = int(num_frames)
        out[video_name] = {
            "timestamps": timestamps,
            "sentences": sentences,
            "fps": fps,
            "num_frames": num_frames,
            "duration": duration,
            "video_duration": duration,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected grouped TACoS payload with top-level dict")

    converted = convert_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"wrote {output_path} with {len(converted)} videos")


if __name__ == "__main__":
    main()
