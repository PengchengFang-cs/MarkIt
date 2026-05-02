import argparse
import json
import math
import re


def cal_iou(seg_a, seg_b):
    max0 = max(seg_a[0], seg_b[0])
    min0 = min(seg_a[0], seg_b[0])
    max1 = max(seg_a[1], seg_b[1])
    min1 = min(seg_a[1], seg_b[1])
    intersection = max(min1 - max0, 0)
    union = max1 - min0
    return intersection / union if union > 0 else 0.0


def extract_pred_span(response_text: str) -> tuple[int, int] | None:
    text = str(response_text)
    frame_match = re.search(
        r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*to\s*(?:frame\s*)?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if frame_match:
        start_value = float(frame_match.group(1))
        end_value = float(frame_match.group(2))
        start_frame = max(0, int(math.floor(start_value)))
        end_frame = max(start_frame, int(math.ceil(end_value)))
        return start_frame, end_frame

    second_match = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds?",
        text,
        re.IGNORECASE,
    )
    if second_match:
        start_value = float(second_match.group(1))
        end_value = float(second_match.group(2))
        start_frame = max(0, int(math.floor(start_value)))
        end_frame = max(start_frame, int(math.ceil(end_value)))
        return start_frame, end_frame

    return None


def get_prediction(item: dict) -> tuple[int, int, int, int]:
    parsed = extract_pred_span(item.get("response", ""))
    if parsed is not None:
        pred_start, pred_end = parsed
    else:
        pred_start = item.get("pred_start", 0)
        pred_end = item.get("pred_end", pred_start)
        pred_start = max(0, int(pred_start))
        pred_end = max(pred_start, int(pred_end))

    stride = int(item.get("frame_stride", 2))
    return pred_start, pred_end, pred_start * stride, pred_end * stride


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
    parser.add_argument("--no_write_back", action="store_true")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        pred_start, pred_end, scaled_start, scaled_end = get_prediction(item)
        item["pred_start"] = pred_start
        item["pred_end"] = pred_end
        item["iou"] = cal_iou([scaled_start, scaled_end], [item["gt_start"], item["gt_end"]])

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
