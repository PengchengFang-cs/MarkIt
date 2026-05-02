import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.video_utils import VideoMetadata

from qwen_vl_utils import process_vision_info


torch.backends.cuda.matmul.allow_tf32 = True

FONT_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "fonts" / "DejaVuSans-Bold.ttf")
DEFAULT_INPUT_FORMAT = (
    "During which frames can we see {}? Answer in the format: 'From Frame x to Frame y'."
)
DEFAULT_NUMBER_INSTRUCTION = "The black numbers on each frame represent the frame number."
DEFAULT_COLOR_NAME_MAP = {
    1: "red",
    2: "yellow",
    3: "blue",
    4: "green",
    5: "cyan",
    6: "magenta",
    7: "orange",
    8: "pink",
    9: "white",
}
IMPLEMENTATION_VERSION = "legacy_noavoid_numpro_fallback_v1"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def ensure_video_name(video_name: str, suffix: str = ".mp4") -> str:
    if str(video_name).endswith(suffix):
        return str(video_name)
    return f"{video_name}{suffix}"


def coerce_float(value) -> float:
    return float(value)


def flatten_grouped_video_dataset(payload: dict) -> list[dict]:
    records = []
    for video_id, sample in payload.items():
        sentences = sample.get("sentences")
        timestamps = sample.get("timestamps")
        duration = sample.get("video_duration", sample.get("duration"))

        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            raise ValueError(f"{video_id}: expected list fields 'sentences' and 'timestamps'.")
        if len(sentences) != len(timestamps):
            raise ValueError(
                f"{video_id}: sentence count {len(sentences)} does not match timestamp count {len(timestamps)}."
            )
        if duration is None:
            raise ValueError(f"{video_id}: missing 'video_duration'/'duration'.")

        nouns_payload = sample.get("nouns")
        for idx, (query, timestamp) in enumerate(zip(sentences, timestamps), start=1):
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"{video_id}: invalid timestamp at index {idx}: {timestamp}")
            if isinstance(nouns_payload, list) and len(nouns_payload) >= idx:
                nouns = nouns_payload[idx - 1]
            elif isinstance(nouns_payload, dict):
                nouns = nouns_payload
            else:
                nouns = {}
            start_time, end_time = timestamp
            records.append(
                {
                    "id": f"{Path(str(video_id)).stem}_{idx}",
                    "video": ensure_video_name(video_id),
                    "start_time": coerce_float(start_time),
                    "end_time": coerce_float(end_time),
                    "query": normalize_text(query),
                    "duration": coerce_float(duration),
                    "nouns": nouns if isinstance(nouns, dict) else {},
                }
            )
    return records


def flatten_list_video_dataset(payload: list[dict]) -> list[dict]:
    records = []
    for idx, sample in enumerate(payload, start=1):
        query = sample.get("query", sample.get("sentence"))
        if query is None:
            raise ValueError(f"List sample at index {idx} is missing 'query'/'sentence'.")

        if "start_time" in sample and "end_time" in sample:
            start_time = sample["start_time"]
            end_time = sample["end_time"]
        else:
            timestamp = sample.get("timestamp", sample.get("timestamps"))
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"List sample at index {idx} is missing valid timestamp fields.")
            start_time, end_time = timestamp

        duration = sample.get("duration", sample.get("video_duration"))
        if duration is None:
            raise ValueError(f"List sample at index {idx} is missing 'duration'/'video_duration'.")

        video_name = sample.get("video", sample.get("video_id", sample.get("clip_id")))
        if video_name is None:
            raise ValueError(f"List sample at index {idx} is missing 'video'/'video_id'/'clip_id'.")

        record_id = sample.get("id", f"{Path(str(video_name)).stem}_{idx}")
        records.append(
            {
                "id": str(record_id),
                "video": ensure_video_name(str(video_name)),
                "start_time": coerce_float(start_time),
                "end_time": coerce_float(end_time),
                "query": normalize_text(query),
                "duration": coerce_float(duration),
                "nouns": sample.get("nouns", {}) if isinstance(sample.get("nouns", {}), dict) else {},
            }
        )
    return records


def load_test_records(testset_path: str) -> list[dict]:
    with open(testset_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return flatten_grouped_video_dataset(payload)
    if isinstance(payload, list):
        return flatten_list_video_dataset(payload)
    raise TypeError(f"Unsupported root JSON type: {type(payload).__name__}")


def _corner_xy(width: int, height: int, box_w: int, box_h: int, position: str, margin: int) -> tuple[int, int]:
    if position == "top_left":
        return margin, margin
    if position == "top_right":
        return width - box_w - margin, margin
    if position == "bottom_left":
        return margin, height - box_h - margin
    if position == "bottom_right":
        return width - box_w - margin, height - box_h - margin
    raise ValueError(f"Unsupported corner position: {position}")


def annotate_frame_with_pil(
    frame: np.ndarray,
    text: str,
    position: str,
    font_size: int,
    color: str = "black",
) -> np.ndarray:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)

    margin = 10
    padding = 2
    stroke_width = 0
    stroke_fill = (255, 255, 255, 255)
    text_fill = (0, 0, 0, 255)

    try:
        font = ImageFont.truetype(FONT_PATH, int(font_size))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + 2 * padding
    box_h = text_h + 2 * padding
    width, height = img.size

    if position == "center":
        x = max(0, min((width - box_w) // 2, width - box_w))
        y = max(0, min((height - box_h) // 2, height - box_h))
    else:
        x, y = _corner_xy(width, height, box_w, box_h, position, margin)

    draw.text(
        (x + padding, y + padding),
        text,
        font=font,
        fill=text_fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    out_rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def annotate_and_save_markit_video(
    file_path: str,
    output_file_path: str,
    position: str,
    font_size: int,
    color: str,
    frame_stride: int,
) -> float:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 1.0
    sample_interval = max(1, int(frame_stride))
    output_fps = max(0.5, fps / sample_interval)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file_path, fourcc, output_fps, (336, 336))

    frame_count = 0
    frame_number = 0
    last_written_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            frame_small = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_NEAREST)
            frame_anno = annotate_frame_with_pil(
                frame_small,
                str(frame_number),
                position,
                font_size,
                color,
            )
            out.write(frame_anno)
            last_written_frame = frame_anno
            frame_number += 1
        frame_count += 1

    if frame_number == 0:
        cap.release()
        out.release()
        raise RuntimeError(f"MarkIt rendering produced zero frames: {file_path}")

    # Very short overlays can collapse to a single frame after sub-sampling,
    # so duplicate the only rendered frame to preserve video semantics.
    if frame_number == 1 and last_written_frame is not None:
        out.write(last_written_frame)

    cap.release()
    out.release()
    return output_fps


def annotate_and_save_numpro_video(
    file_path: str,
    output_file_path: str,
    position: str,
    font_size: int,
    color: str,
) -> float:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {file_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 1.0

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = annotate_frame_with_pil(frame, str(frame_number), position, font_size, color)
        out.write(annotated)
        frame_number += 1

    cap.release()
    out.release()
    return float(fps)


def make_instruction_with_colors(video_info: dict, base_instruction: str) -> str:
    nouns_dict = video_info.get("nouns", {})
    if not isinstance(nouns_dict, dict) or not nouns_dict:
        return base_instruction

    parts = []
    for noun_id, noun in sorted(nouns_dict.items(), key=lambda item: int(item[0])):
        color_name = DEFAULT_COLOR_NAME_MAP.get(int(noun_id), f"color_{noun_id}")
        parts.append(f"({color_name}): {noun}")
    color_desc = "In each frame, the colored masks correspond to the following nouns: " + "; ".join(parts) + "."
    return base_instruction + "\n" + color_desc


def compose_prompt(instruction: str, context: str) -> str:
    if instruction:
        return instruction + "\n" + context
    return context


def sample_video_frames(video_path: str, num_frames: int) -> list[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()

    return [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in frame_indices]


def get_video_total_frames(video_path: str) -> int | None:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return total_frames if total_frames > 0 else None
    finally:
        cap.release()


def build_video_metadata(video_path: str, sampled_frame_count: int) -> VideoMetadata | None:
    if sampled_frame_count <= 0:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None

        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if total_frames <= 0:
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 1.0

        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()

    if total_frames <= sampled_frame_count:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=sampled_frame_count, dtype=int).tolist()

    return VideoMetadata(
        total_num_frames=total_frames,
        fps=fps,
        width=width if width > 0 else None,
        height=height if height > 0 else None,
        duration=total_frames / fps if fps > 0 else None,
        video_backend="decord",
        frames_indices=frame_indices,
    )


def internvl_frame_to_tensor(frame: Image.Image, image_size: int = 448) -> torch.Tensor:
    resized = frame.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def extract_pred_span(response_text: str) -> tuple[int, int]:
    text = normalize_text(response_text)
    frame_match = re.search(
        r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*to\s*(?:frame\s*)?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if frame_match:
        start_value = float(frame_match.group(1))
        end_value = float(frame_match.group(2))
        start_frame = max(0, int(np.floor(start_value)))
        end_frame = max(start_frame, int(np.ceil(end_value)))
        return start_frame, end_frame

    second_match = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds?",
        text,
        re.IGNORECASE,
    )
    if second_match:
        start_value = float(second_match.group(1))
        end_value = float(second_match.group(2))
        start_frame = max(0, int(np.floor(start_value)))
        end_frame = max(start_frame, int(np.ceil(end_value)))
        return start_frame, end_frame

    return 0, 0


class Qwen25VLBackend:
    def __init__(self, model_path: str):
        self.device = "cuda"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.inference_mode()
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_fps: float,
        num_frames: int,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": video_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class InternVL3Backend:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()
        self.device = self.model.device

    @torch.inference_mode()
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_fps: float,
        num_frames: int,
        max_new_tokens: int,
    ) -> str:
        del video_fps
        frames = sample_video_frames(video_path, num_frames)
        pixel_values = torch.stack([internvl_frame_to_tensor(frame) for frame in frames]).to(
            dtype=torch.bfloat16,
            device=self.device,
        )
        num_patches_list = [1] * len(frames)
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )


class VideoLLaMA3Backend:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_fps: float,
        num_frames: int,
        max_new_tokens: int,
    ) -> str:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {"video_path": video_path, "fps": video_fps, "max_frames": num_frames},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


class Qwen3VLBackend:
    def __init__(self, model_path: str):
        from transformers import Qwen3VLForConditionalGeneration

        self.device = "cuda"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.inference_mode()
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_fps: float,
        num_frames: int,
        max_new_tokens: int,
    ) -> str:
        del video_fps
        total_frames = get_video_total_frames(video_path)
        effective_frames = num_frames
        if total_frames is not None and total_frames > 0:
            effective_frames = min(num_frames, total_frames)
            effective_frames = max(2, (effective_frames // 2) * 2)
        template_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        sampling_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": effective_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(sampling_messages)
        if not video_inputs:
            raise RuntimeError(f"Failed to decode video inputs for Qwen3-VL: {video_path}")

        video_metadata = build_video_metadata(video_path, int(video_inputs[0].shape[0]))
        processor_kwargs = dict(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = [video_metadata]
            processor_kwargs["do_sample_frames"] = False

        inputs = self.processor(**processor_kwargs)
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class Qwen2VLBackend:
    def __init__(self, model_path: str):
        self.device = "cuda"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            image_processor.max_pixels = 336 * 336

    @torch.inference_mode()
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_fps: float,
        num_frames: int,
        max_new_tokens: int,
    ) -> str:
        del video_fps
        total_frames = get_video_total_frames(video_path)
        if total_frames is None:
            safe_num_frames = num_frames
        elif total_frames < 2:
            safe_num_frames = total_frames
        else:
            safe_num_frames = min(num_frames, total_frames)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 1,
                        "max_frames": safe_num_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def build_backend(name: str, model_path: str):
    if name == "qwen2_vl":
        return Qwen2VLBackend(model_path)
    if name == "qwen2_5_vl":
        return Qwen25VLBackend(model_path)
    if name == "qwen3_vl":
        return Qwen3VLBackend(model_path)
    if name == "internvl3":
        return InternVL3Backend(model_path)
    if name == "videollama3":
        return VideoLLaMA3Backend(model_path)
    raise ValueError(f"Unsupported backend: {name}")


def atomic_write_json(obj, save_path: str) -> None:
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f"{save_path_obj.name}.",
        suffix=".tmp",
        dir=save_path_obj.parent,
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path_obj)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def load_existing_responses(save_path: str) -> list[dict]:
    if not os.path.exists(save_path):
        return []
    with open(save_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    if responses and any(item.get("impl_version") != IMPLEMENTATION_VERSION for item in responses):
        raise ValueError(
            f"{save_path} contains results from a different markit implementation; use a new save_path."
        )
    return responses


def resolve_raw_video_path(raw_video_root: str, record: dict) -> str:
    if not raw_video_root:
        raise FileNotFoundError(
            f"Missing overlay video for {record['id']} and no --raw_video_root was provided."
        )
    raw_video_path = os.path.join(raw_video_root, record["video"])
    if not os.path.exists(raw_video_path):
        raise FileNotFoundError(f"Missing fallback raw video: {raw_video_path}")
    return raw_video_path


def process_dataset(args):
    backend = build_backend(args.backend, args.model_path)
    records = load_test_records(args.test_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    responses = load_existing_responses(args.save_path)
    processed_ids = {item["id"] for item in responses}

    os.makedirs(args.temp_dir_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"{args.backend}_markit_", dir=args.temp_dir_root)
    try:
        for record in tqdm(records):
            if record["id"] in processed_ids:
                continue

            overlay_path = os.path.join(args.overlay_root, f"{record['id']}_overlay.mp4")
            if os.path.exists(overlay_path):
                try:
                    render_mode = "markit"
                    source_video_path = overlay_path
                    annotated_video_path = os.path.join(temp_dir, f"{record['id']}_markit.mp4")
                    output_fps = annotate_and_save_markit_video(
                        file_path=source_video_path,
                        output_file_path=annotated_video_path,
                        position=args.position,
                        font_size=args.font_size,
                        color=args.color,
                        frame_stride=args.markit_frame_stride,
                    )
                    instruction = make_instruction_with_colors(record, args.number_instruction)
                    frame_stride = args.markit_frame_stride
                except RuntimeError as exc:
                    print(
                        f"[warn] markit overlay failed for {record['id']}: {exc}. "
                        "Falling back to raw-video numbering."
                    )
                    render_mode = "numpro_fallback"
                    source_video_path = resolve_raw_video_path(args.raw_video_root, record)
                    annotated_video_path = os.path.join(temp_dir, f"{record['id']}_numpro.mp4")
                    output_fps = annotate_and_save_numpro_video(
                        file_path=source_video_path,
                        output_file_path=annotated_video_path,
                        position=args.position,
                        font_size=args.font_size,
                        color=args.color,
                    )
                    instruction = args.number_instruction
                    frame_stride = 1
            else:
                render_mode = "numpro_fallback"
                source_video_path = resolve_raw_video_path(args.raw_video_root, record)
                annotated_video_path = os.path.join(temp_dir, f"{record['id']}_numpro.mp4")
                output_fps = annotate_and_save_numpro_video(
                    file_path=source_video_path,
                    output_file_path=annotated_video_path,
                    position=args.position,
                    font_size=args.font_size,
                    color=args.color,
                )
                instruction = args.number_instruction
                frame_stride = 1

            prompt = compose_prompt(instruction, args.input_format.format(record["query"]))
            response_text = backend.generate(
                video_path=annotated_video_path,
                prompt=prompt,
                video_fps=output_fps,
                num_frames=args.num_sampled_frames,
                max_new_tokens=args.max_new_tokens,
            )
            pred_start, pred_end = extract_pred_span(response_text)

            responses.append(
                {
                    "id": record["id"],
                    "video": record["video"],
                    "query": record["query"],
                    "protocol": "markit",
                    "backend": args.backend,
                    "impl_version": IMPLEMENTATION_VERSION,
                    "render_mode": render_mode,
                    "frame_stride": frame_stride,
                    "response": response_text,
                    "gt_start": record["start_time"],
                    "gt_end": record["end_time"],
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "duration": record["duration"],
                }
            )
            atomic_write_json(responses, args.save_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "internvl3", "videollama3"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--overlay_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--raw_video_root", type=str, default="")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--num_sampled_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--input_format", type=str, default=DEFAULT_INPUT_FORMAT)
    parser.add_argument("--number_instruction", type=str, default=DEFAULT_NUMBER_INSTRUCTION)
    parser.add_argument("--position", type=str, default="bottom_right")
    parser.add_argument("--font_size", type=int, default=38)
    parser.add_argument("--color", type=str, default="black")
    parser.add_argument("--markit_frame_stride", type=int, default=2)
    parser.add_argument(
        "--temp_dir_root",
        type=str,
        default="outputs/temp_markit_eval",
    )
    args = parser.parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
