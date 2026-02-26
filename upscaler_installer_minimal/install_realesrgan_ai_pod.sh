#!/usr/bin/env bash
set -euo pipefail

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
err() { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

BASE_DIR="${BASE_DIR:-/workspace/tools/realesrgan_ai}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/Real-ESRGAN}"
VENV_DIR="${VENV_DIR:-${BASE_DIR}/venv}"
STATUS_DIR="${STATUS_DIR:-/workspace/status}"
READY_FLAG="${READY_FLAG:-${STATUS_DIR}/upscaler_ai_ready}"
REPO_URL="${REPO_URL:-https://github.com/xinntao/Real-ESRGAN.git}"

mkdir -p "$BASE_DIR" "$STATUS_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_system_cmd() {
  local cmd="$1"
  local apt_pkg="$2"

  if need_cmd "$cmd"; then
    return 0
  fi

  if ! need_cmd apt-get; then
    err "Missing command '$cmd' and apt-get is not available"
    exit 1
  fi

  if [[ "${EUID}" -ne 0 ]]; then
    err "Missing command '$cmd'. Run as root so apt can install: $apt_pkg"
    exit 1
  fi

  export DEBIAN_FRONTEND=noninteractive
  log "Installing missing package: $apt_pkg"
  apt-get update -y
  apt-get install -y "$apt_pkg"
}

runtime_ready() {
  [[ -d "$REPO_DIR" && -d "$VENV_DIR" ]] || return 1
  source "${VENV_DIR}/bin/activate"
  python - <<'PY'
import importlib.util
import sys

required = ("torch", "torchvision", "basicsr", "realesrgan")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing:", ",".join(missing))
    sys.exit(1)
PY
}

ensure_system_cmd python3 python3
ensure_system_cmd git git
ensure_system_cmd ffmpeg ffmpeg

if [[ -f "$READY_FLAG" ]] && runtime_ready; then
  log "Upscaler runtime already ready ($READY_FLAG). Nothing to do."
  exit 0
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning Real-ESRGAN repository..."
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  log "Using existing repository: $REPO_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv (system-site-packages to reuse pod Python/CUDA stack)..."
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

log "Updating pip tooling..."
python -m pip install --upgrade pip setuptools wheel

REQ_FILE="${REPO_DIR}/requirements.txt"
REQ_FILTERED="/tmp/realesrgan_requirements_no_torch.txt"
if [[ -f "$REQ_FILE" ]]; then
  # Pod already has CUDA torch/torchvision. Skip those to prevent huge redundant installs.
  awk '!/^[[:space:]]*torch([[:space:]]|$|[<>=!~])|^[[:space:]]*torchvision([[:space:]]|$|[<>=!~])/' "$REQ_FILE" > "$REQ_FILTERED"
  log "Installing Real-ESRGAN python requirements (without torch/torchvision)..."
  # Reuse preinstalled CUDA torch stack; avoid slow build-isolation env that re-downloads torch.
  python -m pip install --no-build-isolation -r "$REQ_FILTERED"
else
  warn "requirements.txt missing in repo, continuing with editable install only"
fi

log "Installing Real-ESRGAN package..."
cd "$REPO_DIR"
python -m pip install --no-deps -e .

log "Applying torchvision compatibility shim if required..."
python - <<'PY'
from pathlib import Path
import torchvision

transforms_dir = Path(torchvision.__file__).resolve().parent / "transforms"
shim = transforms_dir / "functional_tensor.py"
if not shim.exists():
    shim.write_text(
        "# Auto-generated compatibility shim for BasicSR on newer torchvision\n"
        "from .functional import *\n",
        encoding="utf-8",
    )
    print("Created shim:", shim)
else:
    print("Shim already exists:", shim)
PY

log "Running runtime checks..."
python - <<'PY'
import torch
import torchvision
import basicsr
import realesrgan

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
print("basicsr:", basicsr.__version__ if hasattr(basicsr, "__version__") else "ok")
print("realesrgan:", realesrgan.__version__ if hasattr(realesrgan, "__version__") else "ok")
PY

touch "$READY_FLAG"
log "Upscaler AI runtime is ready."
log "Ready flag: $READY_FLAG"
log "Run command:"
printf '%s\n' "bash /workspace/upscaler_installer_minimal/upscale_video_ai_cuda.sh --in /workspace/edit.mp4 --out /workspace/output_tiktok_1080x1920.mp4 --model x2 --target 1080x1920 --tile 0"
