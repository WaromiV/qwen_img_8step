FROM runpod/worker-comfyui:5.5.1-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DISABLE_REQUIRE=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    COMFYUI_DIR=/comfyui \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    MODEL_ROOT=/runpod-volume/huggingface/qwen-image-edit-2511-gguf \
    PRELOAD_MODEL=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt && \
    rm -rf /comfyui/custom_nodes/ComfyUI-GGUF && \
    git clone --depth=1 --branch feat_optimized_dequant https://github.com/blepping/ComfyUI-GGUF /comfyui/custom_nodes/ComfyUI-GGUF && \
    pip install -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt

COPY handler.py /app/handler.py
COPY app /app/app
COPY workflow /app/workflow
COPY README.md /app/README.md
COPY test_input.json /app/test_input.json
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["python3", "-u", "handler.py"]
