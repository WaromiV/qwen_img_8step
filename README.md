# Qwen Image Edit 2511 RunPod Worker

Queue-based RunPod Serverless worker for `Qwen/Qwen-Image-Edit-2511`.

## Repo Layout

```text
.
├── app/
│   ├── __init__.py
│   └── logic.py
├── .dockerignore
├── .gitignore
├── .runpod/
│   ├── hub.json
│   └── tests.json
├── Dockerfile
├── handler.py
├── README.md
├── requirements.txt
└── test_input.json
```

## Features

- Standard `runpod.serverless.start({"handler": handler})` worker
- Supports single-image and multi-image editing
- Accepts `image_base64`, `image_url`, `image_path`, or `images`
- Returns base64 output in JSON
- Uses `/runpod-volume/huggingface` automatically when a network volume is mounted
- CPU offloading is disabled and the pipeline is hardcoded to run on GPU only

## Input Shape

```json
{
  "input": {
    "prompt": "Turn this product shot into a polished industrial concept render.",
    "image_base64": "<base64>",
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "width": 1024,
    "height": 1024,
    "seed": 0,
    "output_format": "png"
  }
}
```

For multi-image editing:

```json
{
  "input": {
    "prompt": "Place these two characters into a shared cinematic scene.",
    "images": [
      {"image_base64": "<base64-1>"},
      {"image_base64": "<base64-2>"}
    ]
  }
}
```

## Output Shape

```json
{
  "ok": true,
  "model_id": "Qwen/Qwen-Image-Edit-2511",
  "offload_mode": "model",
  "seed": 0,
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "width": 1024,
  "height": 1024,
  "mime_type": "image/png",
  "image_base64": "<base64>"
}
```

## Runtime Environment

- `MODEL_ID` defaults to `Qwen/Qwen-Image-Edit-2511`
- `OFFLOAD_MODE` is effectively hardcoded to `none` in code; CPU offloading is disabled
- `PRELOAD_MODEL=1` preloads on worker startup; default is `0`
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` should point at `/runpod-volume/...` when using a network volume
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the Docker image to reduce fragmentation issues

## Recommended RunPod Endpoint Settings

- Endpoint type: `Queue`
- Container disk: at least `40 GB`
- Network volume: attach one and mount it
- `HF_TOKEN`: required if model access is gated in your account context
- Use a GPU tier with enough VRAM because CPU offloading is disabled

## Local Test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
python handler.py
```

## Notes

- This repo uses `git+https://github.com/huggingface/diffusers` because Qwen image editing support lands there first
- `torchvision` is required because the Qwen image edit stack pulls in `Qwen2VLVideoProcessor`
- `true_cfg_scale` is the main guidance knob for Qwen image edit; pass `negative_prompt` as well
- For fast Hub/GitHub smoke tests, `.runpod/tests.json` uses a lightweight `self_test`
