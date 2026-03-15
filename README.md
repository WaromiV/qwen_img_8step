# Qwen Image Edit 2511 GGUF RunPod Worker

Queue-based RunPod Serverless worker for `unsloth/Qwen-Image-Edit-2511-GGUF` using the `qwen-image-edit-2511-Q2_K.gguf` variant through headless ComfyUI + ComfyUI-GGUF.

The image build is trimmed for RunPod's builder by starting from `runpod/worker-comfyui:5.5.1-base` instead of installing ComfyUI from scratch.

The Docker image also sets `NVIDIA_DISABLE_REQUIRE=1` to bypass the base image's strict CUDA requirement gate during container init on RunPod hosts that otherwise reject startup before the worker code even runs.

## What Changed

This repo no longer uses Diffusers.

- Runtime stack: `ComfyUI` + `ComfyUI-GGUF`
- Quantized model: `qwen-image-edit-2511-Q2_K.gguf`
- Text encoder: `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- VAE: `qwen_image_vae.safetensors`
- Base image: `runpod/worker-comfyui:5.5.1-base`
- Custom node source: `blepping/ComfyUI-GGUF` `feat_optimized_dequant`

## Repo Layout

```text
.
├── app/
│   ├── __init__.py
│   └── logic.py
├── workflow/
│   └── qwen_image_edit_2511_gguf_single.json
├── .dockerignore
├── .gitignore
├── .runpod/
│   ├── hub.json
│   └── tests.json
├── Dockerfile
├── handler.py
├── README.md
├── requirements.txt
├── start.sh
└── test_input.json
```

## Required Model Files

The worker downloads and links these automatically:

- `https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q2_K.gguf`
- `https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors`

If `/runpod-volume` is mounted, the worker auto-caches these under `/runpod-volume/huggingface/qwen-image-edit-2511-gguf` and links them into `/comfyui/models/...`.

## Input Shape

```json
{
  "input": {
    "prompt": "Turn this concept sketch into a polished industrial render.",
    "image_base64": "<base64>",
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0,
    "width": 1024,
    "height": 1024,
    "seed": 0
  }
}
```

Accepted image inputs:

- `image_base64`
- `image_url`
- `image_path`
- `images` (only the first image is used in the current single-image workflow)

## Output Shape

```json
{
  "ok": true,
  "model_id": "unsloth/Qwen-Image-Edit-2511-GGUF::qwen-image-edit-2511-Q2_K.gguf",
  "runtime": "comfyui-gguf",
  "seed": 0,
  "num_inference_steps": 20,
  "true_cfg_scale": 4.0,
  "width": 1024,
  "height": 1024,
  "mime_type": "image/png",
  "image_base64": "<base64>",
  "prompt_id": "<comfy-prompt-id>",
  "output_filename": "<saved-file>"
}
```

## Recommended RunPod Settings

- Endpoint type: `Queue`
- GPU: `A40` or stronger
- Container disk: `40 GB+`
- Network volume: attach one if you want persistent downloaded models; default cache path is `/runpod-volume/huggingface/qwen-image-edit-2511-gguf`
- `PRELOAD_MODEL=0` for easier startup; set `1` only if you intentionally keep workers warm

## Performance Notes

- `PRELOAD_MODEL` now defaults to `1` so ComfyUI is started before the first job instead of inside request execution
- ComfyUI now starts with `--highvram`
- The workflow now uses `UnetLoaderGGUFAdvanced` with `optimize=triton` and `dequant_dtype=float16`
- This is aimed at reducing the dequantization overhead that makes `Q2_K` look much slower than its file size suggests

## Pod Volume Notes

- The worker expects the RunPod network volume mount at `/runpod-volume`
- Model cache defaults to `/runpod-volume/huggingface/qwen-image-edit-2511-gguf`
- Hugging Face cache envs also point at `/runpod-volume/huggingface`
- If the volume is not mounted, the worker falls back to `/opt/model-cache/qwen-image-edit-2511-gguf`

## Notes

- This repo mandates the `Q2_K` GGUF variant for maximum compression
- Current workflow is single-image edit only
- It uses ComfyUI native Qwen edit nodes and swaps the stock model loader for `UnetLoaderGGUF`

Release trigger note: this line exists to force a fresh RunPod rebuild when needed.

Release trigger note 2: another no-op line for a fresh rebuild.
