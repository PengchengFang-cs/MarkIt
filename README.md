# MarkIt: Training-Free Visual Markers for Precise Video Temporal Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2604.25886-b31b1b.svg)](https://arxiv.org/abs/2604.25886)

Pengcheng Fang, Yuxia Chen, Xiaohao Cai

Paper: [arXiv:2604.25886](https://arxiv.org/abs/2604.25886) | [PDF](https://arxiv.org/pdf/2604.25886)

MarkIt is a training-free visual prompting pipeline for video temporal grounding. It builds on NumPro by combining frame-number prompts with object-centric visual marks: target nouns are rendered as color-coded masks on video frames, and the model is instructed to localize the queried moment by returning frame indices.

This repository contains a public, lightweight code release. It does not include datasets, model checkpoints, generated masks, overlay videos, logs, or result files.

## Method Overview

For each video-query pair, MarkIt follows this workflow:

1. Prepare query nouns, for example `person`, `broom`, or `floor`.
2. Render color-coded masks or outlines for the noun regions in each frame.
3. Add visible black frame numbers to the rendered video.
4. Prompt a video-language model with both the frame-number instruction and the noun-color mapping.
5. Parse the model response in the format `From Frame x to Frame y`.
6. Convert frame predictions back to timestamps and evaluate mIoU / recall.

## Main Files

- `eval/activitynet_n5_test_font_find_new.py`: original LongVA MarkIt evaluation entry for ActivityNet-style data.
- `eval/charades_n5_test_font_find_new.py`: original LongVA MarkIt evaluation entry for Charades/ActivityNet-style data.
- `eval/vlm_mr_markit.py`: recommended multi-backend MarkIt evaluation entry.
- `eval/vlm_mr_markit_noannot.py`: evaluates already-rendered overlay videos without adding frame numbers.
- `eval/vlm_mr_markit_noannot_2s.py`: no-annotation evaluation with 2-second preprocessing.
- `eval/metrics_mr_markit_noannot.py`: maps frame predictions to timestamps and reports moment-retrieval metrics.
- `longva/`: LongVA code inherited from the NumPro implementation.
- `tools/`: small dataset-conversion helpers for TACoS-style experiments.

## Installation

```bash
conda create -n markit python=3.10 -y
conda activate markit
pip install -r requirements.txt
```

Install model-specific dependencies as required by the backend you use. Some models may require compatible CUDA, FlashAttention, or `trust_remote_code=True` support from Hugging Face.

## Data Format

The evaluation scripts accept either a list of records:

```json
[
  {
    "id": "video_001_1",
    "video": "video_001.mp4",
    "query": "a person opens a door",
    "start_time": 12.0,
    "end_time": 18.0,
    "duration": 60.0,
    "nouns": {"1": "person", "2": "door"}
  }
]
```

or a grouped dataset with `sentences`, `timestamps`, `duration` / `video_duration`, and optional `nouns`.

Overlay videos are expected to be named:

```text
<id>_overlay.mp4
```

Optional masks are expected to be named:

```text
<id>_mask.npy
```

## Recommended Evaluation

Run from the repository root.

```bash
python eval/vlm_mr_markit.py \
  --backend qwen2_5_vl \
  --model_path /path/to/model \
  --overlay_root /path/to/markit_overlay_videos \
  --raw_video_root /path/to/raw_videos \
  --test_path /path/to/test.json \
  --save_path outputs/markit_predictions.json \
  --max_samples -1
```

Supported backends in `eval/vlm_mr_markit.py`:

```text
qwen2_vl, qwen2_5_vl, qwen3_vl, internvl3, videollama3
```

Evaluate predictions:

```bash
python eval/metrics_mr_markit_noannot.py \
  --json_file outputs/markit_predictions.json \
  --overlay_root /path/to/markit_overlay_videos \
  --raw_video_root /path/to/raw_videos
```

## Original LongVA Entries

ActivityNet-style:

```bash
python eval/activitynet_n5_test_font_find_new.py \
  --data_path /path/to/markit_overlay_videos \
  --mask_path /path/to/markit_masks \
  --test_path /path/to/activitynet_test.json \
  --save_path outputs/activitynet_markit_longva.json \
  --model_path lmms-lab/LongVA-7B-DPO
```

Charades-style:

```bash
python eval/charades_n5_test_font_find_new.py \
  --data_path /path/to/markit_overlay_videos \
  --test_path /path/to/charades_test.json \
  --save_path outputs/charades_markit_longva.json \
  --model_path lmms-lab/LongVA-7B-DPO
```

## Acknowledgement

This codebase is based on NumPro:

- Paper: `Number it: Temporal Grounding Videos like Flipping Manga`
- Repository: `https://github.com/yongliang-wu/NumPro`

We thank the NumPro authors and the LongVA ecosystem for their excellent work.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{fang2026markit,
  title={MarkIt: Training-Free Visual Markers for Precise Video Temporal Grounding},
  author={Fang, Pengcheng and Chen, Yuxia and Cai, Xiaohao},
  journal={arXiv preprint arXiv:2604.25886},
  year={2026}
}
```
