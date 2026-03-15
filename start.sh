#!/usr/bin/env bash
set -euo pipefail

cd "${COMFYUI_DIR:-/comfyui}"
exec python -u main.py --disable-auto-launch --disable-metadata --highvram --listen "${COMFY_HOST:-127.0.0.1}" --port "${COMFY_PORT:-8188}" --log-stdout
