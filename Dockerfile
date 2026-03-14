FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    OFFLOAD_MODE=model \
    PRELOAD_MODEL=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu124 torch && \
    pip install -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY app /app/app
COPY README.md /app/README.md
COPY test_input.json /app/test_input.json

CMD ["python3", "-u", "handler.py"]
