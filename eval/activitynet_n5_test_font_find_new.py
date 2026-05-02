import os
import sys

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
# 项目根目录：.../donkey_place
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import copy
import warnings
import json
import os
import multiprocessing
from typing import List, Optional, Union
import numpy as np
import torch
# from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoConfig
from transformers import TextIteratorStreamer
from threading import Thread
from tqdm import tqdm

from peft import PeftModel
from datetime import timedelta

import cv2

torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")

from longva.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from longva.conversation import conv_templates, SeparatorStyle
from longva.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from longva.model.builder import load_pretrained_model
import re
import cv2
from PIL import Image
import shutil

ID2COLOR_NAME = {
    1: "red",        # 原来写的是 red，其实画出来是蓝
    2: "yellow",        # 原来 yellow，其实更接近青色
    3: "blue",      # 0,128,255 视觉上偏橙红
    4: "green",
    5: "cyan",      # 原来写 cyan，实际是黄
    6: "magenta",
    7: "orange",  # 天空蓝 / 青蓝
    8: "pink",
    9: "white",
}

def _integral_image(binary_01: np.ndarray) -> np.ndarray:
    # binary_01: uint8/0-1, shape(H,W)
    return np.cumsum(np.cumsum(binary_01, axis=0), axis=1)

def _rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    # sum over [y0:y1, x0:x1), with integral image ii
    # clamp
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = max(0, x1); y1 = max(0, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    A = ii[y1-1, x1-1]
    B = ii[y0-1, x1-1] if y0 > 0 else 0
    C = ii[y1-1, x0-1] if x0 > 0 else 0
    D = ii[y0-1, x0-1] if (y0 > 0 and x0 > 0) else 0
    return int(A - B - C + D)

def _find_free_xy(avoid_mask: np.ndarray, box_w: int, box_h: int,
                  preferred: str = "bottom_right",
                  margin: int = 10, grid_step: int = 8) -> tuple[int, int] | None:
    """
    avoid_mask: (H,W) bool/0-1; True 表示禁止覆盖区域（mask区域）
    返回一个 (x,y)，使得 [x:x+box_w, y:y+box_h] 内 avoid_mask 全为 False
    """
    H, W = avoid_mask.shape
    if box_w > W or box_h > H:
        return None

    # 候选角落：先试 preferred，再试其他
    corners = {
        "top_left": (margin, margin),
        "top_right": (W - box_w - margin, margin),
        "bottom_left": (margin, H - box_h - margin),
        "bottom_right": (W - box_w - margin, H - box_h - margin),
        "center": ((W - box_w)//2, (H - box_h)//2),
    }
    order = [preferred] + [k for k in ["top_left","top_right","bottom_left","bottom_right","center"] if k != preferred]

    ii = _integral_image(avoid_mask.astype(np.uint8))

    def ok(x, y):
        return _rect_sum(ii, x, y, x + box_w, y + box_h) == 0

    for k in order:
        x, y = corners[k]
        x = max(0, min(int(x), W - box_w))
        y = max(0, min(int(y), H - box_h))
        if ok(x, y):
            return x, y

    # 网格搜索：从边缘到中心扫（更容易找到空白背景）
    for y in range(margin, H - box_h - margin + 1, grid_step):
        for x in range(margin, W - box_w - margin + 1, grid_step):
            if ok(x, y):
                return x, y

    return None
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"
    
    
def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    model = PeftModel.from_pretrained(model, lora_path)
    return model


class LongVA:
    """
    LongVA Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/LongVA-7B-DPO",
        lora_path: Optional[str] = None,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "average",
        token_strategy: Optional[
            str
        ] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame in the context
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        model_name = (
            model_name
            if model_name is not None
            else get_model_name_from_path(pretrained)
        )

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained, # model path
                    None, # model base
                    model_name, # model name
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )
            if lora_path:
                print("Adding LoRA to the model in LongVA class")
                self._model = load_lora(self._model, lora_path)
                self._model.merge_and_unload()
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )


            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._word_size = 1

        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)
    
    def load_video_all(self, video_path):

        if isinstance(video_path, str):
            cv2_vr = cv2.VideoCapture(video_path)
        else:
            cv2_vr = cv2.VideoCapture(video_path[0])
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

        if duration > 240:
            frame_indices = np.linspace(0, duration - 1, 240, dtype=int)
        else:
            frame_indices = np.arange(duration)

        video_data = []
        for frame_idx in frame_indices:
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f"video error at {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            video_data.append(image)
            
        cv2_vr.release()
        return video_data

    def stream_generate_until(self, requests: dict, gen_kwargs: dict) -> List[str]:

        question_input = []
        instruction = requests["instruction"]
        visuals = requests["visuals"]
        context = requests["context"]
        task_type = requests["task_type"]
        
        if task_type == "text":
            image_tensor = None
        # encode, pad, and truncate contexts for this batch
        elif task_type == "image":  # For image task
            image_tensor = process_images(visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device)
                    for _image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        elif task_type == "video":  # For video task
            image_tensor = []
            max_frames = gen_kwargs.get("sample_frames", self.max_frames_num)
            if "sample_frames" in gen_kwargs:
                gen_kwargs.pop("sample_frames")

            try:
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visuals, max_frames)
                elif self.video_decode_backend == "all":
                    frames = self.load_video_all(visuals)

                frames = (
                    self._image_processor.preprocess(frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .to(self.device)
                )
                image_tensor.append(frames)
            except Exception as e:
                image_tensor = None

            task_type = "video"

        if (
            image_tensor is not None
            and len(image_tensor) != 0
            and DEFAULT_IMAGE_TOKEN not in context
        ):
            """
            Three scenarios:
            1. No image, and there for, no image token should be added.
            2. image token is already specified in the context, so we don't need to add it.
            3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
            4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
            """
            if task_type == "image":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(visuals)
                    if isinstance(visuals, list)
                    else [DEFAULT_IMAGE_TOKEN]
                )
            elif task_type == "video":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(frames)
                    if self.token_strategy == "multiple"
                    else [DEFAULT_IMAGE_TOKEN]
                )

            image_tokens = " ".join(image_tokens)
            question = image_tokens + "\n" + instruction + context
        else:
            question = context

        # This is much safer for llama3, as we now have some object type in it
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        for prev_conv in requests["prev_conv"]:
            conv.append_message(conv.roles[0], prev_conv[0])
            conv.append_message(conv.roles[1], prev_conv[1])

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()
        question_input.append(prompt_question)

        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for prompt in question_input
        ]
        pad_token_ids = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = self.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_ids
        ).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        if task_type == "image":
            gen_kwargs["image_sizes"] = [
                visual.size for visual in visuals
            ]  # (width, height)
        elif task_type == "video":
            gen_kwargs["modalities"] = ["video"]
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: attention to this major generation step...
        if "image_aspect_ratio" in gen_kwargs.keys():
            gen_kwargs.pop("image_aspect_ratio")

        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        num_image_tokens = (
            question.count(DEFAULT_IMAGE_TOKEN)
            * self.model.get_vision_tower().num_patches
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        gen_kwargs["max_new_tokens"] = min(
            gen_kwargs["max_new_tokens"],
            max_context_length - input_ids.shape[-1] - num_image_tokens,
        )

        if gen_kwargs["max_new_tokens"] < 1:
            yield json.dumps(
                {
                    "text": question
                    + "Exceeds max token length. Please start a new conversation, thanks.",
                    "error_code": 0,
                }
            ).encode() + b"\0"
            return

        try:
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    inputs=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    use_cache=self.use_cache,
                    streamer=streamer,
                    **gen_kwargs,
                ),
            )
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[: -len(stop_str)]
                yield json.dumps(
                    {"text": generated_text, "error_code": 0}
                ).encode() + b"\0"
        except Exception as e:
            raise e
           

def _corner_xy(W, H, box_w, box_h, corner, margin):
    if corner == "top_left":
        x, y = margin, margin
    elif corner == "top_right":
        x, y = W - box_w - margin, margin
    elif corner == "bottom_left":
        x, y = margin, H - box_h - margin
    elif corner == "bottom_right":
        x, y = W - box_w - margin, H - box_h - margin
    else:
        raise ValueError("Invalid corner")
    x = max(0, min(int(x), W - box_w))
    y = max(0, min(int(y), H - box_h))
    return x, y

def _mask_overlap_count(mask_bool, x, y, w, h):
    # mask_bool: HxW bool
    return int(np.count_nonzero(mask_bool[y:y+h, x:x+w]))


# full
def annotate_frame_with_pil(frame, text, position, font_size, color=None, avoid_mask=None):
    """
    - 不画背景（无白底）
    - 强制黑色字体
    - avoid_mask=None: 按 position 固定画
    - avoid_mask!=None: 只在四角里选遮盖最少；若 overlap==0，优先级：右下 > 左下 > 右上 > 左上
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)

    font_path = str(PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf")

    margin = 10
    padding = 2
    stroke_width = 0
    stroke_fill = (255, 255, 255, 255)

    
    # text_fill = (255, 0, 0, 255)      # 红
    # text_fill = (0, 200, 255, 255)
    # text_fill = (255, 0, 255, 255)  # 亮紫
    # text_fill = (255, 170, 0, 255)  # 橙黄
    text_fill = (0, 0, 0, 255)  # 强制黑色


    W, H = img.size

    # 字体（只用一次，不缩字号）
    try:
        font = ImageFont.truetype(font_path, int(font_size))
    except Exception:
        font = ImageFont.load_default()

    # 计算文字占位框尺寸（用于与 mask 求重叠）
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + 2 * padding
    box_h = text_h + 2 * padding

    if avoid_mask is None:
        # 沿用你原来的固定位置逻辑
        if position in ("top_left", "top_right", "bottom_left", "bottom_right"):
            x, y = _corner_xy(W, H, box_w, box_h, position, margin)
        elif position == "center":
            x, y = (W - box_w) // 2, (H - box_h) // 2
            x = max(0, min(int(x), W - box_w))
            y = max(0, min(int(y), H - box_h))
        else:
            raise ValueError("Invalid position argument")
    else:
        # 四角候选 + 你指定的优先级顺序
        corners = ["bottom_right", "bottom_left", "top_right", "top_left"]

        scored = []
        for idx, c in enumerate(corners):
            cx, cy = _corner_xy(W, H, box_w, box_h, c, margin)
            ov = _mask_overlap_count(avoid_mask.astype(bool), cx, cy, box_w, box_h)
            scored.append((ov, idx, cx, cy))

        # 先选 overlap 最小，再按优先级 idx 最小（即右下优先）
        ov, _, x, y = min(scored, key=lambda t: (t[0], t[1]))

    # 不画背景，只画字
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

# # only 不膨胀
# def annotate_frame_with_pil_(frame, text, position, font_size, color=None, avoid_mask=None):
#     """
#     - 不画背景（无白底）
#     - 强制黑色字体
#     - 避免遮盖 mask：仍用 _find_free_xy 搜索；找不到就缩字号；仍找不到就不画
#     - 与原版相比：不膨胀 mask（不做 dilate）
#     """
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img_rgb).convert("RGBA")
#     draw = ImageDraw.Draw(img)

#     font_path = str(PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf")

#     margin = 10
#     padding = 2
#     stroke_width = 0
#     stroke_fill = (255, 255, 255, 255)

#     text_fill = (0, 0, 0, 255)  # 强制黑色字体（RGBA）
#     W, H = img.size

#     # ✅ 不膨胀：原来这里的 dilate 逻辑删掉，只做 bool 化
#     if avoid_mask is not None:
#         avoid_mask = avoid_mask.astype(bool)

#     for fs in range(int(font_size), 11, -2):
#         try:
#             font = ImageFont.truetype(font_path, fs)
#         except Exception:
#             font = ImageFont.load_default()

#         bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
#         text_w = bbox[2] - bbox[0]
#         text_h = bbox[3] - bbox[1]

#         box_w = text_w + 2 * padding
#         box_h = text_h + 2 * padding

#         if avoid_mask is None:
#             if position == "top_left":
#                 x, y = margin, margin
#             elif position == "top_right":
#                 x, y = W - box_w - margin, margin
#             elif position == "bottom_left":
#                 x, y = margin, H - box_h - margin
#             elif position == "bottom_right":
#                 x, y = W - box_w - margin, H - box_h - margin
#             elif position == "center":
#                 x, y = (W - box_w) // 2, (H - box_h) // 2
#             else:
#                 raise ValueError("Invalid position argument")
#             x = max(0, min(int(x), W - box_w))
#             y = max(0, min(int(y), H - box_h))
#         else:
#             xy = _find_free_xy(avoid_mask, box_w, box_h, preferred=position, margin=margin, grid_step=8)
#             if xy is None:
#                 continue
#             x, y = xy

#         draw.text(
#             (x + padding, y + padding),
#             text,
#             font=font,
#             fill=text_fill,
#             stroke_width=stroke_width,
#             stroke_fill=stroke_fill,
#         )

#         out_rgb = np.array(img.convert("RGB"))
#         return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

#     return frame

# # 不膨胀不缩字号
# def annotate_frame_with_pil(frame, text, position, font_size, color=None, avoid_mask=None):
#     """
#     B (必画版):
#     - 不画背景（无白底）
#     - 强制黑色字体
#     - 不膨胀 mask
#     - 不缩字号：只用 font_size
#     - 选位置方法与原版一致：
#         * avoid_mask=None: 按 position 固定位置
#         * avoid_mask!=None: 先用 _find_free_xy 搜索
#           若搜索失败(None) -> 退化为按 position 固定位置（保证必画）
#     """
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img_rgb).convert("RGBA")
#     draw = ImageDraw.Draw(img)

#     font_path = str(PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf")

#     margin = 10
#     padding = 2
#     stroke_width = 0
#     stroke_fill = (255, 255, 255, 255)

#     text_fill = (0, 0, 0, 255)  # 强制黑色
#     W, H = img.size

#     # 不膨胀：只做 bool 化
#     if avoid_mask is not None:
#         avoid_mask = avoid_mask.astype(bool)

#     # 不缩字号：只用一次 font_size
#     try:
#         font = ImageFont.truetype(font_path, int(font_size))
#     except Exception:
#         font = ImageFont.load_default()

#     bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
#     text_w = bbox[2] - bbox[0]
#     text_h = bbox[3] - bbox[1]
#     box_w = text_w + 2 * padding
#     box_h = text_h + 2 * padding

#     # 先按原逻辑决定 x,y
#     if avoid_mask is None:
#         xy = None
#     else:
#         xy = _find_free_xy(avoid_mask, box_w, box_h, preferred=position, margin=margin, grid_step=8)

#     if xy is not None:
#         x, y = xy
#     else:
#         # 搜索失败 or 没有 mask：退化为固定位置（和你原版 avoid_mask=None 分支一致）
#         if position == "top_left":
#             x, y = margin, margin
#         elif position == "top_right":
#             x, y = W - box_w - margin, margin
#         elif position == "bottom_left":
#             x, y = margin, H - box_h - margin
#         elif position == "bottom_right":
#             x, y = W - box_w - margin, H - box_h - margin
#         elif position == "center":
#             x, y = (W - box_w) // 2, (H - box_h) // 2
#         else:
#             raise ValueError("Invalid position argument")

#         x = max(0, min(int(x), W - box_w))
#         y = max(0, min(int(y), H - box_h))

#     # 不画背景，只画字
#     draw.text(
#         (x + padding, y + padding),
#         text,
#         font=font,
#         fill=text_fill,
#         stroke_width=stroke_width,
#         stroke_fill=stroke_fill,
#     )

#     out_rgb = np.array(img.convert("RGB"))
#     return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def annotate_and_save_video(file_path, output_file_path, position, font_size, color, mask_npy_path=None):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {file_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = int(fps * 2)  # 你原来的逻辑

    mask_stack = None
    if mask_npy_path is not None and os.path.exists(mask_npy_path):
        mask_stack = np.load(mask_npy_path)  # (T,H,W) uint8
    # 注意：这里假设 frame_number（抽帧计数）与 mask_stack 的时间维一致
    # 若你的 overlay 视频本身就是 1fps 采样后视频，建议 sample_interval=1，且 frame_number==mask_index

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, 0.5, (336, 336))

    frame_count = 0
    frame_number = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            avoid_mask = None
            # if mask_stack is not None and frame_number < len(mask_stack):
            #     H, W = 336,336
            #     m = cv2.resize(mask_stack[frame_number], (W, H), interpolation=cv2.INTER_NEAREST)
            #     avoid_mask = (m != 0)

            frame_small = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_NEAREST)    
            # 叠字（避免遮盖 mask）
            frame_anno = annotate_frame_with_pil(
                frame_small, str(frame_number), position, font_size, color, avoid_mask=avoid_mask
            )

            out.write(frame_anno)
            frame_number += 1

        frame_count += 1

    cap.release()
    out.release()

        
def make_instruction_with_colors(video_info: dict, base_instruction: str) -> str:
    """
    video_info 里有:
    - "nouns": {"1": "person", "2": "broom", "3": "floor", ...}
    """

    nouns_dict = video_info.get("nouns", {})
    if not isinstance(nouns_dict, dict) or not nouns_dict:
        return None  # 没有 nouns 就保持原样

    # 按 id 排序，确保 1,2,3,... 的顺序
    items = sorted(nouns_dict.items(), key=lambda kv: int(kv[0]))
    # parts = [noun for _, noun in items]
    # if parts:
    #     color_desc = (
    #         "In each frame, the red masks correspond to the following nouns: "
    #         + "; ".join(parts)
    #         + "."
    #     )
    # return "In each frame, red regions highlight the objects mentioned in the query."
    
    # 拼一个描述：id1(red): person; id2(yellow): broom; ...
    parts = []
    for id_str, noun in items:
        cid = int(id_str)
        color_name = ID2COLOR_NAME.get(cid, f"color_{cid}")
        parts.append(f"({color_name}): {noun}")

    '''
    1.
    color_desc = (
        "In each frame, the following nouns are indicated by color-coded masks. "
        "Each object is marked by a colored region overlaid on the image, together "
        "with a clearly visible outline in the same color. The outline serves as a "
        "strong visual cue for identifying the object: "
        + "; ".join(parts) + "."
    )

    2.
    color_desc = "In each frame, the colored masks correspond to the following nouns: " + "; ".join(parts) + "."

    3. 
    color_desc = (
        "In each frame, the colored masks correspond to the following nouns. "
        "Each mask includes a filled region and a matching colored outline that "
        "helps visually distinguish the object: "
        + "; ".join(parts) + "."
    )

    '''
    
    color_desc = "In each frame, the colored masks correspond to the following nouns: " + "; ".join(parts) + "."

    # 合并到最终 instruction
    return base_instruction + "\n" + color_desc

    # return color_desc

# def video_demo(model, video_info, data_path, num_sampled_frames, input_format, instruction, position="bottom_right", font_size=40, color="red", mask_path=None, temp_dir="temp"):
#     # visual_path = os.path.join(data_path, video_info["video"])
#     overlay_name = f"{video_info['id']}_overlay.mp4"
#     mask_npy_name = f"{video_info['id']}_mask.npy"
#     visual_path = os.path.join(data_path, overlay_name)
#     mask_npy_path = os.path.join(mask_path, mask_npy_name)
#     if not os.path.exists(visual_path):
#         print(f"❌ 视频不存在: {visual_path}, use orignal video")
#         sys.exit()
    
#     annotated_video_path = os.path.join(temp_dir, f"{video_info['id']}_{os.getpid()}.mp4")
#     # annotated_video_path = os.path.join(temp_dir, video_info['video'])
#     annotate_and_save_video(visual_path, annotated_video_path, position, font_size, color, mask_npy_path=mask_npy_path)
    
#     input_visuals = [annotated_video_path]
#     # The frame number is indicated on each frame. 
#     # input_format = "Find the video segment that corresponds to the given textual query '{}' and determine its start and end timestamps."
#     input_context = input_format.format(video_info["query"])
#     instruction = make_instruction_with_colors(video_info,instruction)
#     print(f"{video_info['id']}:",instruction)

#     task_type = "video"
#     gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": num_sampled_frames}
#     query = {
#         "instruction": instruction,
#         "visuals": input_visuals,
#         "context": input_context,
#         "task_type": task_type,
#         "prev_conv": [],
#     }
#     response = {"id": video_info["id"], "response": "", "gt_start": video_info["start_time"], "gt_end": video_info["end_time"], "pred_start": 0, "pred_end": 0, "duration": video_info["duration"]}
#     try:
#         for x in model.stream_generate_until(query, gen_kwargs):
#             output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
#             response["response"] = output
#     except Exception as e:
#         print(e)
#     match = re.search(r"from\s*(?:frame\s*)?(\d+)\s*to\s*(?:frame\s*)?(\d+)", response['response'], re.IGNORECASE)
#     if match:
#         response["pred_start"] = int(match.group(1))
#         response["pred_end"] = int(match.group(2))
    
#     return response

def video_demo(
    model,
    video_info,
    data_path,
    num_sampled_frames,
    input_format,
    instruction,
    position="bottom_right",
    font_size=40,
    color="red",
    mask_path=None,
    temp_dir="temp",
):
    """
    1) 从 overlay 视频生成带帧号的临时视频 temp/<id>_<pid>.mp4
    2) 用该临时视频做推理
    3) 无论成功/失败，最后都会删除临时视频
    """
    print(position,font_size)
    overlay_name = f"{video_info['id']}_overlay.mp4"
    mask_npy_name = f"{video_info['id']}_mask.npy"

    visual_path = os.path.join(data_path, overlay_name)
    mask_npy_path = os.path.join(mask_path, mask_npy_name) if mask_path is not None else None

    if not os.path.exists(visual_path):
        print(f"❌ 视频不存在: {visual_path}, use original video")
        sys.exit()
    
    if not os.path.exists(mask_npy_path):
        print(f"❌ 视频不存在: {mask_npy_path}, use original video")
        sys.exit()

    # 生成临时视频
    annotated_video_path = os.path.join(temp_dir, f"{video_info['id']}_{os.getpid()}.mp4")
    annotate_and_save_video(
        visual_path,
        annotated_video_path,
        position,
        font_size,
        color,
        mask_npy_path=mask_npy_path,
    )

    try:
        # 准备输入
        input_visuals = [annotated_video_path]
        input_context = input_format.format(video_info["query"])
        full_instruction = make_instruction_with_colors(video_info, instruction)
        print(f"{video_info['id']}:", full_instruction)

        task_type = "video"
        gen_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0,
            "do_sample": False,
            "sample_frames": num_sampled_frames,
        }
        query = {
            "instruction": full_instruction,
            "visuals": input_visuals,
            "context": input_context,
            "task_type": task_type,
            "prev_conv": [],
        }

        response = {
            "id": video_info["id"],
            "response": "",
            "gt_start": video_info["start_time"],
            "gt_end": video_info["end_time"],
            "pred_start": 0,
            "pred_end": 0,
            "duration": video_info["duration"],
        }

        # 推理（流式）
        try:
            for x in model.stream_generate_until(query, gen_kwargs):
                output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                response["response"] = output
        except Exception as e:
            print(e)

        # 解析输出
        match = re.search(
            r"from\s*(?:frame\s*)?(\d+)\s*to\s*(?:frame\s*)?(\d+)",
            response["response"],
            re.IGNORECASE,
        )
        if match:
            response["pred_start"] = int(match.group(1))
            response["pred_end"] = int(match.group(2))

        return response

    finally:
        # ✅ 无论成功失败都清理临时视频
        try:
            if os.path.exists(annotated_video_path):
                os.remove(annotated_video_path)
        except Exception as e:
            print(f"[WARN] Failed to remove temp video: {annotated_video_path}, err={e}")

from pathlib import Path

def atomic_write_json(obj, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, save_path)  # 原子替换（同一文件系统内）

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sampled_frames", type=int, default=32)
    parser.add_argument("--videobackend", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing MarkIt overlay videos named <id>_overlay.mp4.")
    parser.add_argument("--mask_path", type=str, required=True, help="Directory containing MarkIt masks named <id>_mask.npy.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to write JSON predictions.")
    parser.add_argument("--test_path", type=str, required=True, help="ActivityNet-style test annotation JSON.")
    parser.add_argument("--model_path", type=str, default="lmms-lab/LongVA-7B-DPO")
    parser.add_argument("--input_format", type=str, default="During which frames can we see {}? Answer in the format: 'From Frame x to Frame y'.")
    '''
    1. The black numbers on each frame represent the frame number.
    2. The black numbers on each frame indicate the frame number.
    3. The number shown on each frame indicates the frame index.
    4. The black numbers shown in a corner of each frame represent the frame number.
    5. The numbers shown in a corner of each frame represent the frame number.
    '''
    parser.add_argument("--instruction", type=str, default="The black numbers on each frame represent the frame number.")
    parser.add_argument("--lora_path", type=str, default='')
    parser.add_argument("--position", type=str, default="bottom_right")
    parser.add_argument("--font_size", type=int, default=38)
    parser.add_argument("--color", type=str, default="red")
    parser.add_argument("--temp_dir", type=str, default="temp")
    
    args = parser.parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)
    videobackend = args.videobackend
    data_path = args.data_path
    model_path = args.model_path
    input_format = args.input_format
    instruction = args.instruction
    num_sampled_frames = args.num_sampled_frames
    position = args.position
    font_size = args.font_size
    color = args.color
    model = LongVA(pretrained=model_path, lora_path=args.lora_path, model_name="llava_qwen", device=args.device, device_map=args.device, video_decode_backend=videobackend)
    
    testset_path = args.test_path
    save_path = args.save_path
        
    with open(testset_path, 'r') as f:
        video_list = json.load(f)
    
    responses = []
    if os.path.exists(save_path):
        responses = json.load(open(save_path, 'r'))
    processed_ids = {response["id"] for response in responses}
    save_every = 50
    print('charades_LongVA_n5_alpha0.3_binarycolors_instruction_outline3_P2.json,是charades内部和边缘都为彩色。P2')
    for i, video_info in enumerate(tqdm(video_list), 1):
        if video_info["id"] in processed_ids:
            continue
        
        response = video_demo(model, video_info, data_path, num_sampled_frames, input_format, instruction, position, font_size, color,mask_path=args.mask_path)
        responses.append(response)
        if i % save_every == 0:
            atomic_write_json(responses, save_path)

    # 循环结束再强制保存一次
    atomic_write_json(responses, save_path)
