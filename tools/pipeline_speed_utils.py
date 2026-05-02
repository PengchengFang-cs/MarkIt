from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont


DEFAULT_FONT_PATH = Path(__file__).resolve().parents[1] / "DejaVuSans-Bold.ttf"


@lru_cache(maxsize=64)
def get_cached_font(font_size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont:
    resolved_font_path = font_path or str(DEFAULT_FONT_PATH)
    return ImageFont.truetype(resolved_font_path, font_size)


def read_sampled_frames_once(video_path: str, indices: np.ndarray) -> dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"frames": [], "indices": np.array([], dtype=int), "width": 0, "height": 0}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    unique_indices = np.unique(np.asarray(indices, dtype=int))
    if unique_indices.size == 0 or width <= 0 or height <= 0:
        cap.release()
        return {"frames": [], "indices": np.array([], dtype=int), "width": width, "height": height}

    frames: list[np.ndarray] = []
    kept_indices: list[int] = []
    target_pos = 0
    current_target = int(unique_indices[target_pos])
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == current_target:
            frames.append(frame.copy())
            kept_indices.append(current_target)
            target_pos += 1
            if target_pos >= len(unique_indices):
                break
            current_target = int(unique_indices[target_pos])
        frame_idx += 1

    cap.release()
    return {
        "frames": frames,
        "indices": np.asarray(kept_indices, dtype=int),
        "width": width,
        "height": height,
    }


def save_frames_array(path: str, frames: list[np.ndarray] | np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(frames, dtype=np.uint8), allow_pickle=False)


def load_frames_array(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def _ceil_by_factor(number: float, factor: int) -> int:
    import math
    return int(math.ceil(number / factor) * factor)


def _floor_by_factor(number: float, factor: int) -> int:
    import math
    return int(math.floor(number / factor) * factor)


def get_sample_indices_by_target_fps(
    total_frames: int,
    video_fps: float,
    target_fps: float,
    min_frames: int = 4,
    max_frames: int = 768,
    frame_factor: int = 2,
) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    min_frames = _ceil_by_factor(min_frames, frame_factor)
    max_frames = _floor_by_factor(min(max_frames, total_frames), frame_factor)
    nframes = total_frames / max(video_fps, 1e-6) * target_fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = _round_by_factor(nframes, frame_factor)
    nframes = max(frame_factor, min(nframes, total_frames))
    idx = np.linspace(0, total_frames - 1, int(nframes)).round().astype(int)
    return np.clip(idx, 0, total_frames - 1)


def sample_video_frames_by_target_fps(
    video_path: str,
    target_fps: float,
    min_frames: int = 4,
    max_frames: int = 768,
    frame_factor: int = 2,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0,), dtype=np.uint8)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    cap.release()
    indices = get_sample_indices_by_target_fps(
        total_frames,
        video_fps,
        target_fps,
        min_frames=min_frames,
        max_frames=max_frames,
        frame_factor=frame_factor,
    )
    sampled = read_sampled_frames_once(video_path, indices)
    return np.asarray(sampled["frames"], dtype=np.uint8)


def truncate_nouns_dict(nouns: dict[str, str], max_nouns: int | None) -> dict[str, str]:
    if not isinstance(nouns, dict):
        return {}
    if max_nouns is None or max_nouns <= 0:
        return {str(k): str(v) for k, v in nouns.items()}
    ordered_items = sorted(nouns.items(), key=lambda item: int(item[0]))
    kept_items = ordered_items[:max_nouns]
    return {str(k): str(v) for k, v in kept_items}
