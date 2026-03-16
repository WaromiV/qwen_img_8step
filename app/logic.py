import base64
import copy
import datetime as dt
import io
import json
import os
import resource
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests
from PIL import Image

RUNPOD_VOLUME = Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))
COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/comfyui"))
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = Path(
    os.getenv("WORKFLOW_PATH", "/app/workflow/02_qwen_Image_edit_subgraphed.json")
)
DIFF_MODEL_FILENAME = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
LORA_FILENAME = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
TEXT_ENCODER_FILENAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_FILENAME = "qwen_image_vae.safetensors"
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_TRUE_CFG_SCALE = float(os.getenv("DEFAULT_TRUE_CFG_SCALE", "4.0"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))

if RUNPOD_VOLUME.exists():
    MODEL_ROOT = Path(
        os.getenv(
            "MODEL_ROOT",
            str(RUNPOD_VOLUME / "huggingface" / "qwen-image-edit-2511-gguf"),
        )
    )
else:
    MODEL_ROOT = Path(
        os.getenv("MODEL_ROOT", "/opt/model-cache/qwen-image-edit-2511-gguf")
    )

MODEL_SPECS = [
    {
        "filename": DIFF_MODEL_FILENAME,
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "subdir": "diffusion_models",
    },
    {
        "filename": TEXT_ENCODER_FILENAME,
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "subdir": "text_encoders",
    },
    {
        "filename": VAE_FILENAME,
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "subdir": "vae",
    },
    {
        "filename": LORA_FILENAME,
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "subdir": "loras",
    },
]

BOOT_MONO = time.monotonic()
START_LOCK = threading.Lock()
SERVER_PROCESS: Optional[subprocess.Popen] = None


def _iso_timestamp() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe_disk_usage(path: Path):
    try:
        total, used, free = shutil.disk_usage(path)
        return {"path": str(path), "total": total, "used": used, "free": free}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}


def _proc_status() -> dict:
    data = {
        "pid": os.getpid(),
        "threads": None,
        "vmrss_kb": None,
        "vmhwm_kb": None,
        "loadavg": None,
        "ru_maxrss_kb": None,
    }
    try:
        load1, load5, load15 = os.getloadavg()
        data["loadavg"] = [load1, load5, load15]
    except OSError:
        pass

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        data["ru_maxrss_kb"] = usage.ru_maxrss
    except Exception:  # noqa: BLE001
        pass

    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(errors="ignore").splitlines():
            if line.startswith("Threads:"):
                data["threads"] = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                data["vmrss_kb"] = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                data["vmhwm_kb"] = int(line.split()[1])
    return data


def _gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "utilization_gpu_pct": float(parts[4]),
                "temperature_c": float(parts[5]),
                "power_draw_w": float(parts[6]),
            }
        )
    return gpus


def _resource_snapshot() -> dict:
    paths = [Path("/"), Path("/tmp"), COMFYUI_DIR, MODEL_ROOT]
    if RUNPOD_VOLUME.exists():
        paths.append(RUNPOD_VOLUME)

    snapshot = {
        "process": _proc_status(),
        "disk": [],
        "gpu": _gpu_snapshot(),
    }
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        snapshot["disk"].append(_safe_disk_usage(path))
    return snapshot


def _model_file_state() -> list[dict]:
    state = []
    for spec in MODEL_SPECS:
        path = MODEL_ROOT / spec["filename"]
        state.append(
            {
                "filename": spec["filename"],
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
            }
        )
    return state


def log_event(
    stage: str, *, request_id=None, include_resources=False, **fields
) -> None:
    payload = {
        "ts": _iso_timestamp(),
        "mono_s": round(time.monotonic() - BOOT_MONO, 3),
        "stage": stage,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    payload.update(fields)
    if include_resources:
        payload["resources"] = _resource_snapshot()
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def _download_headers() -> dict:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "opencode-qwen-worker/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _download_file(url: str, destination: Path, *, request_id=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    started = time.monotonic()
    bytes_written = 0
    log_event(
        "model.download.start",
        request_id=request_id,
        url=url,
        destination=str(destination),
        include_resources=True,
    )
    with requests.get(
        url,
        stream=True,
        timeout=600,
        headers=_download_headers(),
    ) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bytes_written += len(chunk)
    temp_path.replace(destination)
    log_event(
        "model.download.done",
        request_id=request_id,
        url=url,
        destination=str(destination),
        bytes_written=bytes_written,
        elapsed_s=round(time.monotonic() - started, 3),
        include_resources=True,
    )


def _link_or_copy(source: Path, target: Path, *, request_id=None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        log_event(
            "model.link.skip",
            request_id=request_id,
            source=str(source),
            target=str(target),
        )
        return
    try:
        target.symlink_to(source)
        mode = "symlink"
    except OSError:
        target.write_bytes(source.read_bytes())
        mode = "copy"
    log_event(
        "model.link.done",
        request_id=request_id,
        source=str(source),
        target=str(target),
        mode=mode,
    )


def ensure_model_files(*, request_id=None) -> None:
    log_event(
        "model.ensure.start",
        request_id=request_id,
        model_root=str(MODEL_ROOT),
        model_files=_model_file_state(),
        include_resources=True,
    )
    for spec in MODEL_SPECS:
        source = MODEL_ROOT / spec["filename"]
        if not source.exists():
            _download_file(spec["url"], source, request_id=request_id)
        else:
            log_event(
                "model.ensure.hit",
                request_id=request_id,
                filename=spec["filename"],
                path=str(source),
                size=source.stat().st_size,
            )
        target = COMFYUI_DIR / "models" / spec["subdir"] / spec["filename"]
        _link_or_copy(source, target, request_id=request_id)
    log_event(
        "model.ensure.done",
        request_id=request_id,
        model_files=_model_file_state(),
        include_resources=True,
    )


def _log_path() -> Path:
    return Path(os.getenv("COMFY_LOG_PATH", "/tmp/comfyui-qwen.log"))


def _start_command() -> list[str]:
    return ["/app/start.sh"]


def _server_alive() -> bool:
    global SERVER_PROCESS
    return SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None


def _wait_until_ready(timeout: int = 300, *, request_id=None) -> None:
    deadline = time.monotonic() + timeout
    attempts = 0
    last_error = None
    while time.monotonic() < deadline:
        attempts += 1
        try:
            response = requests.get(f"{COMFY_BASE_URL}/system_stats", timeout=5)
            if response.ok:
                data = response.json()
                log_event(
                    "comfy.ready",
                    request_id=request_id,
                    attempts=attempts,
                    system_stats=data,
                    include_resources=True,
                )
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempts == 1 or attempts % 5 == 0:
            log_event(
                "comfy.wait",
                request_id=request_id,
                attempts=attempts,
                last_error=last_error,
                include_resources=True,
            )
        time.sleep(2)

    log_path = _log_path()
    tail = ""
    if log_path.exists():
        tail = log_path.read_text(errors="ignore")[-4000:]
    raise RuntimeError(f"ComfyUI did not become ready: {last_error}\n{tail}")


def warmup_model(*, request_id=None) -> None:
    global SERVER_PROCESS
    with START_LOCK:
        if _server_alive():
            comfy_pid = SERVER_PROCESS.pid if SERVER_PROCESS else None
            log_event(
                "warmup.skip",
                request_id=request_id,
                reason="server_already_alive",
                comfy_pid=comfy_pid,
                include_resources=True,
            )
            return

        warmup_started = time.monotonic()
        log_event(
            "warmup.start",
            request_id=request_id,
            comfy_dir=str(COMFYUI_DIR),
            comfy_base_url=COMFY_BASE_URL,
            workflow_path=str(WORKFLOW_PATH),
            include_resources=True,
        )
        ensure_model_files(request_id=request_id)
        log_path = _log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_event(
            "comfy.process.spawn",
            request_id=request_id,
            command=_start_command(),
            cwd=str(COMFYUI_DIR),
            log_path=str(log_path),
        )
        with log_path.open("ab") as log_file:
            SERVER_PROCESS = subprocess.Popen(  # noqa: S603
                _start_command(),
                cwd=COMFYUI_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        comfy_pid_runtime = SERVER_PROCESS.pid
        log_event(
            "comfy.process.spawned",
            request_id=request_id,
            comfy_pid=comfy_pid_runtime,
        )
        _wait_until_ready(request_id=request_id)
        log_event(
            "warmup.done",
            request_id=request_id,
            elapsed_s=round(time.monotonic() - warmup_started, 3),
            comfy_pid=comfy_pid_runtime,
            include_resources=True,
        )


def _decode_image(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _load_image(job_input: dict, *, request_id=None) -> Image.Image:
    source_kind = None
    if job_input.get("image_base64"):
        raw = _decode_image(job_input["image_base64"])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        source_kind = "image_base64"
    elif job_input.get("image_url"):
        response = requests.get(job_input["image_url"], timeout=120)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        source_kind = "image_url"
    elif job_input.get("image_path"):
        image = Image.open(job_input["image_path"]).convert("RGB")
        source_kind = "image_path"
    elif job_input.get("images"):
        first = job_input["images"][0]
        if isinstance(first, str):
            image = Image.open(io.BytesIO(_decode_image(first))).convert("RGB")
            source_kind = "images[0]:string"
        elif first.get("image_base64"):
            image = Image.open(
                io.BytesIO(_decode_image(first["image_base64"]))
            ).convert("RGB")
            source_kind = "images[0].image_base64"
        elif first.get("image_url"):
            response = requests.get(first["image_url"], timeout=120)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            source_kind = "images[0].image_url"
        elif first.get("image_path"):
            image = Image.open(first["image_path"]).convert("RGB")
            source_kind = "images[0].image_path"
        else:
            raise ValueError("First images[] item is missing image data")
    else:
        raise ValueError("Provide image_base64, image_url, image_path, or images")

    log_event(
        "request.image.loaded",
        request_id=request_id,
        source_kind=source_kind,
        image_mode=image.mode,
        image_size=list(image.size),
    )
    return image


def _write_input_image(image: Image.Image, job_id: str, *, request_id=None) -> str:
    filename = f"{job_id}.png"
    output_path = COMFYUI_DIR / "input" / filename
    image.save(output_path, format="PNG")
    log_event(
        "request.image.saved",
        request_id=request_id,
        filename=filename,
        path=str(output_path),
        size=output_path.stat().st_size,
    )
    return filename


def _load_workflow(*, request_id=None) -> dict:
    workflow = json.loads(WORKFLOW_PATH.read_text())
    log_event(
        "workflow.loaded",
        request_id=request_id,
        workflow_path=str(WORKFLOW_PATH),
        node_count=len(workflow),
    )
    return workflow


def _patch_workflow(
    workflow: dict,
    *,
    input_filename: str,
    prompt: str,
    seed: int,
    steps: int,
    true_cfg_scale: float,
    width: int,
    height: int,
    filename_prefix: str,
    request_id=None,
) -> dict:
    patched = copy.deepcopy(workflow)
    load_node = None
    k_sampler_node = None
    prompt_nodes = []

    for node in patched.get("nodes", []):
        node_type = node.get("type")
        if node_type == "LoadImage" and load_node is None:
            load_node = node
        elif node_type == "KSampler":
            k_sampler_node = node
        elif node_type == "TextEncodeQwenImageEditPlus":
            prompt_nodes.append(node)

    if load_node and load_node.get("widgets_values"):
        load_node["widgets_values"][0] = input_filename
    if load_node is not None:
        inputs = load_node.setdefault("inputs", [])
        for inp in inputs:
            if inp["name"] == "image":
                inp["value"] = input_filename
                break
        else:
            inputs.append({"name": "image", "value": input_filename})

    for node in prompt_nodes:
        node["widgets_values"] = [prompt]
        inputs = node.setdefault("inputs", [])
        for inp in inputs:
            if inp["name"] == "prompt":
                inp["value"] = prompt
                break
        else:
            inputs.append({"name": "prompt", "value": prompt})

    if k_sampler_node:
        k_sampler_node["widgets_values"] = [
            seed,
            "randomize",
            steps,
            true_cfg_scale,
            "euler",
            "simple",
            1.0,
        ]
        inputs = k_sampler_node.setdefault("inputs", [])
        overrides = [
            ("seed", seed),
            ("steps", steps),
            ("cfg", true_cfg_scale),
            ("sampler_name", "euler"),
            ("scheduler", "simple"),
            ("denoise", 1.0),
        ]
        existing = {inp["name"]: inp for inp in inputs}
        for name, value in overrides:
            if name in existing:
                existing[name]["value"] = value
            else:
                inputs.append({"name": name, "value": value})

    for node in patched.get("nodes", []):
        if node.get("type") == "SaveImage":
            node["widgets_values"] = [filename_prefix]
            inputs = node.setdefault("inputs", [])
            for inp in inputs:
                if inp.get("name") == "filename_prefix":
                    inp["value"] = filename_prefix
                    break
            else:
                inputs.append({"name": "filename_prefix", "value": filename_prefix})
        elif node.get("type") == "LoadImage":
            node["widgets_values"] = [input_filename]
            inputs = node.setdefault("inputs", [])
            for inp in inputs:
                if inp.get("name") == "image":
                    inp["value"] = input_filename
                    break
            else:
                inputs.append({"name": "image", "value": input_filename})

    log_event(
        "workflow.patched",
        request_id=request_id,
        input_filename=input_filename,
        prompt_preview=prompt[:160],
        seed=seed,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        filename_prefix=filename_prefix,
    )
    return patched


def _convert_blueprint_to_prompt_payload(
    workflow: dict, *, input_filename: str | None = None
) -> dict:
    allowed_types = {
        "LoadImage",
        "FluxKontextImageScale",
        "ImageScaleToTotalPixels",
        "UNETLoader",
        "LoraLoaderModelOnly",
        "ModelSamplingAuraFlow",
        "CFGNorm",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImageEditPlus",
        "FluxKontextMultiReferenceLatentMethod",
        "VAEEncode",
        "EmptySD3LatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    links = workflow.get("links", []) or []
    link_map: dict[int, tuple[str, int]] = {}
    allowed_node_ids = {
        str(node["id"])
        for node in workflow.get("nodes", [])
        if node.get("type") in allowed_types
    }
    for entry in links:
        link_id, src_node, src_output, dest_node, _, _ = entry
        if (
            str(src_node) not in allowed_node_ids
            or str(dest_node) not in allowed_node_ids
        ):
            continue
        link_map[link_id] = (str(src_node), src_output)

    prompt_nodes: dict[str, dict] = {}
    for node in workflow.get("nodes", []):
        node_type = node.get("type")
        if node_type not in allowed_types:
            continue
        node_id = str(node["id"])
        converted = {"class_type": node_type, "inputs": {}}
        inputs = converted["inputs"]
        widgets = node.get("widgets_values") or []
        for inp in node.get("inputs", []):
            name = inp.get("name")
            if not name:
                continue
            if "link" in inp:
                link_ref = link_map.get(inp["link"])
                if link_ref:
                    inputs[name] = [link_ref[0], link_ref[1]]
                continue
            if "value" in inp:
                inputs[name] = inp["value"]
            elif "default_value" in inp:
                inputs[name] = inp["default_value"]

        if node_type == "LoadImage":
            target_image = input_filename or (widgets[0] if widgets else None)
            if target_image:
                inputs.setdefault("image", target_image)
        if node_type == "ImageScaleToTotalPixels" and widgets:
            inputs.setdefault("upscale_method", widgets[0])
            if len(widgets) > 1:
                inputs.setdefault("megapixels", widgets[1])
            if len(widgets) > 2:
                inputs.setdefault("helper", widgets[2])
        if node_type == "UNETLoader":
            inputs.setdefault("unet_name", DIFF_MODEL_FILENAME)
            inputs.setdefault("weight_dtype", "fp8_e4m3fn")
        if node_type == "LoraLoaderModelOnly":
            inputs.setdefault("lora_name", LORA_FILENAME)
            inputs.setdefault("strength_model", widgets[1] if len(widgets) > 1 else 1)
        if node_type == "ModelSamplingAuraFlow" and widgets:
            inputs.setdefault("shift", widgets[0])
        if node_type == "CFGNorm" and widgets:
            inputs.setdefault("strength", widgets[0])
        if node_type == "CLIPLoader":
            inputs.setdefault("clip_name", TEXT_ENCODER_FILENAME)
            inputs.setdefault("type", widgets[1] if len(widgets) > 1 else "qwen_image")
            inputs.setdefault("device", "default")
        if node_type == "VAELoader":
            inputs.setdefault("vae_name", widgets[0] if widgets else VAE_FILENAME)
        if node_type == "SaveImage" and widgets:
            inputs.setdefault("filename_prefix", widgets[0])

        prompt_nodes[node_id] = converted
    return prompt_nodes


def _submit_prompt(workflow: dict, *, input_filename: str, request_id=None) -> str:
    prompt_payload = _convert_blueprint_to_prompt_payload(
        workflow, input_filename=input_filename
    )
    payload = {"prompt": prompt_payload, "client_id": str(uuid.uuid4())}
    response = requests.post(f"{COMFY_BASE_URL}/prompt", json=payload, timeout=30)
    if response.status_code >= 400:
        log_event(
            "prompt.submission_failed",
            request_id=request_id,
            status_code=response.status_code,
            response_text=response.text,
        )
    response.raise_for_status()
    data = response.json()
    log_event(
        "prompt.submitted",
        request_id=request_id,
        prompt_id=data["prompt_id"],
        number=data.get("number"),
        node_errors=data.get("node_errors"),
    )
    return data["prompt_id"]


def _poll_history(prompt_id: str, *, timeout: int = 1800, request_id=None) -> dict:
    deadline = time.monotonic() + timeout
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        response = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get(prompt_id):
            history = data[prompt_id]
            log_event(
                "prompt.history.ready",
                request_id=request_id,
                prompt_id=prompt_id,
                polls=polls,
                status=(history.get("status") or {}).get("status_str"),
                completed=(history.get("status") or {}).get("completed"),
            )
            return history
        if polls == 1 or polls % 5 == 0:
            log_event(
                "prompt.history.wait",
                request_id=request_id,
                prompt_id=prompt_id,
                polls=polls,
                include_resources=True,
            )
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for ComfyUI history for prompt_id={prompt_id}"
    )


def _summarize_history_error(history: dict) -> dict:
    status = history.get("status") or {}
    messages = status.get("messages") or []
    last_error = None
    for item in messages:
        if item and item[0] == "execution_error":
            last_error = item[1]
    summary = {
        "status_str": status.get("status_str"),
        "completed": status.get("completed"),
        "messages_count": len(messages),
        "outputs_keys": sorted((history.get("outputs") or {}).keys()),
    }
    if last_error:
        summary.update(
            {
                "node_id": last_error.get("node_id"),
                "node_type": last_error.get("node_type"),
                "exception_type": last_error.get("exception_type"),
                "exception_message": last_error.get("exception_message"),
                "executed_nodes": last_error.get("executed"),
            }
        )
    return summary


def _extract_image_metadata(history: dict, *, request_id=None) -> dict:
    outputs = history.get("outputs") or {}
    images = (outputs.get("15") or {}).get("images") or []
    if not images:
        summary = _summarize_history_error(history)
        log_event(
            "prompt.history.error",
            request_id=request_id,
            summary=summary,
            include_resources=True,
        )
        raise RuntimeError(f"ComfyUI produced no images: {summary}")
    image_meta = images[0]
    log_event(
        "prompt.history.output",
        request_id=request_id,
        image_meta=image_meta,
    )
    return image_meta


def _fetch_output_image(image_meta: dict, *, request_id=None) -> bytes:
    query = urlencode(
        {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
    )
    response = requests.get(f"{COMFY_BASE_URL}/view?{query}", timeout=120)
    response.raise_for_status()
    log_event(
        "prompt.output.fetched",
        request_id=request_id,
        filename=image_meta["filename"],
        bytes=len(response.content),
    )
    return response.content


def edit_image(job_input: dict, *, request_id=None) -> dict:
    started = time.monotonic()
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing required field: prompt")

    log_event(
        "request.start",
        request_id=request_id,
        prompt_len=len(prompt),
        provided_keys=sorted(job_input.keys()),
        include_resources=True,
    )

    warmup_model(request_id=request_id)

    image = _load_image(job_input, request_id=request_id)
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    steps = int(
        job_input.get("num_inference_steps", job_input.get("steps", DEFAULT_STEPS))
    )
    true_cfg_scale = float(job_input.get("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE))
    seed = int(job_input.get("seed", 0))
    job_id = uuid.uuid4().hex
    input_filename = _write_input_image(image, job_id, request_id=request_id)
    workflow = _patch_workflow(
        _load_workflow(request_id=request_id),
        input_filename=input_filename,
        prompt=prompt,
        seed=seed,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        filename_prefix=f"QwenImageEditGGUF_{job_id}",
        request_id=request_id,
    )
    prompt_id = _submit_prompt(
        workflow, input_filename=input_filename, request_id=request_id
    )
    history = _poll_history(prompt_id, request_id=request_id)
    image_meta = _extract_image_metadata(history, request_id=request_id)
    output_bytes = _fetch_output_image(image_meta, request_id=request_id)
    output_image = Image.open(io.BytesIO(output_bytes))

    result = {
        "ok": True,
        "model_id": f"Comfy-Org/Qwen-Image-Edit_ComfyUI::{DIFF_MODEL_FILENAME}",
        "runtime": "comfyui-gguf",
        "seed": seed,
        "num_inference_steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "width": output_image.width,
        "height": output_image.height,
        "mime_type": "image/png",
        "image_base64": base64.b64encode(output_bytes).decode("utf-8"),
        "prompt_id": prompt_id,
        "output_filename": image_meta["filename"],
    }
    log_event(
        "request.done",
        request_id=request_id,
        prompt_id=prompt_id,
        output_filename=image_meta["filename"],
        elapsed_s=round(time.monotonic() - started, 3),
        include_resources=True,
    )
    return result
