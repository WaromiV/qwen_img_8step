import base64
import io
import os
import threading
from pathlib import Path

import requests
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
OFFLOAD_MODE = "none"
RUNPOD_VOLUME = Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))

if RUNPOD_VOLUME.exists():
    CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", RUNPOD_VOLUME / "huggingface"))
else:
    CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/huggingface"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))

DEFAULT_TRUE_CFG_SCALE = float(os.getenv("DEFAULT_TRUE_CFG_SCALE", "4.0"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "40"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this worker.")

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
PIPELINE = None
PIPELINE_LOCK = threading.Lock()


def _enable_memory_helpers(pipe: QwenImageEditPlusPipeline) -> None:
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


def _configure_pipeline(pipe: QwenImageEditPlusPipeline) -> QwenImageEditPlusPipeline:
    pipe.to("cuda")
    _enable_memory_helpers(pipe)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def warmup_model() -> QwenImageEditPlusPipeline:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=str(CACHE_DIR),
        )
        PIPELINE = _configure_pipeline(pipe)
        return PIPELINE


def _decode_image(image_string: str) -> bytes:
    if image_string.startswith("data:"):
        image_string = image_string.split(",", 1)[1]
    return base64.b64decode(image_string)


def _load_single_image(item) -> Image.Image:
    if isinstance(item, str):
        return Image.open(io.BytesIO(_decode_image(item))).convert("RGB")

    if item.get("image_base64"):
        return Image.open(io.BytesIO(_decode_image(item["image_base64"]))).convert(
            "RGB"
        )
    if item.get("image_url"):
        response = requests.get(item["image_url"], timeout=120)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    if item.get("image_path"):
        return Image.open(item["image_path"]).convert("RGB")

    raise ValueError("Each image must provide image_base64, image_url, or image_path")


def _load_images(job_input: dict):
    if job_input.get("images"):
        return [_load_single_image(item) for item in job_input["images"]]
    if job_input.get("image_base64"):
        return [_load_single_image({"image_base64": job_input["image_base64"]})]
    if job_input.get("image_url"):
        return [_load_single_image({"image_url": job_input["image_url"]})]
    if job_input.get("image_path"):
        return [_load_single_image({"image_path": job_input["image_path"]})]
    raise ValueError("Provide image_base64, image_url, image_path, or images")


def _normalize_output_format(output_format: str) -> str:
    fmt = output_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("output_format must be png, jpeg, jpg, or webp")
    return fmt


def edit_image(job_input: dict) -> dict:
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing required field: prompt")

    images = _load_images(job_input)
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    true_cfg_scale = float(job_input.get("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE))
    negative_prompt = str(job_input.get("negative_prompt", " "))
    num_inference_steps = int(
        job_input.get("num_inference_steps", job_input.get("steps", DEFAULT_STEPS))
    )
    guidance_scale = job_input.get("guidance_scale")
    num_images_per_prompt = int(job_input.get("num_images_per_prompt", "1"))
    max_sequence_length = int(
        job_input.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
    )
    output_format = _normalize_output_format(str(job_input.get("output_format", "png")))
    seed = int(job_input.get("seed", 0))

    generator = None if seed < 0 else torch.Generator("cpu").manual_seed(seed)
    pipeline = warmup_model()

    with torch.inference_mode():
        output = pipeline(
            image=images if len(images) > 1 else images[0],
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

    result_image = output.images[0]
    buffer = io.BytesIO()
    save_kwargs = {"quality": 95} if output_format == "JPEG" else {}
    result_image.save(buffer, format=output_format, **save_kwargs)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    mime_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }[output_format]

    return {
        "ok": True,
        "model_id": MODEL_ID,
        "offload_mode": OFFLOAD_MODE,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "width": result_image.width,
        "height": result_image.height,
        "mime_type": mime_type,
        "image_base64": image_base64,
    }
