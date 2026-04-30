#!/bin/bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

duration() {
  local start_ts="$1"
  echo "$(( $(date +%s) - start_ts ))s"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

observed_bytes() {
  local target="$1"
  if [ -e "$target" ]; then
    du -sb "$target" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

run_download_with_guard() {
  local label="$1"
  local target="$2"
  shift 2

  if [ "${INIT_CHECK_ONLY:-0}" = "1" ] || [ "${INIT_SKIP_DOWNLOADS:-0}" = "1" ]; then
    HF_DOWNLOAD_LAST_STATUS="skipped"
    log "[download] CHECK/SKIP: $label -> $target"
    return 0
  fi

  local start_ts
  local pid
  local rc
  local elapsed
  local size
  local last_size="-1"
  local unchanged_for="0"
  start_ts="$(date +%s)"
  log "[download] START: $label"
  log "[download] target=$target timeout=${HF_DOWNLOAD_TIMEOUT_SECONDS}s stall_timeout=${HF_DOWNLOAD_STALL_TIMEOUT_SECONDS}s"

  set +e
  "$@" &
  pid=$!
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep "$HF_DOWNLOAD_PROGRESS_SECONDS"
    elapsed="$(( $(date +%s) - start_ts ))"
    size="$(observed_bytes "$target")"
    if [ "$size" = "$last_size" ]; then
      unchanged_for="$(( unchanged_for + HF_DOWNLOAD_PROGRESS_SECONDS ))"
    else
      unchanged_for="0"
      last_size="$size"
    fi
    log "[download] PROGRESS: $label elapsed=${elapsed}s observed_bytes=${size} unchanged_for=${unchanged_for}s"

    if [ "$elapsed" -ge "$HF_DOWNLOAD_TIMEOUT_SECONDS" ]; then
      log "[download] TIMEOUT: $label exceeded ${HF_DOWNLOAD_TIMEOUT_SECONDS}s"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 5
      kill -9 "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1
      set -e
      return 124
    fi
    if [ "$unchanged_for" -ge "$HF_DOWNLOAD_STALL_TIMEOUT_SECONDS" ]; then
      log "[download] STALLED: $label no byte progress for ${unchanged_for}s"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 5
      kill -9 "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1
      set -e
      return 124
    fi
  done
  wait "$pid"
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    HF_DOWNLOAD_LAST_STATUS="downloaded"
    log "[download] DONE: $label duration=$(duration "$start_ts") observed_bytes=$(observed_bytes "$target")"
  else
    HF_DOWNLOAD_LAST_STATUS="failed"
    log "[download] FAILED: $label rc=$rc duration=$(duration "$start_ts") observed_bytes=$(observed_bytes "$target")"
  fi
  return "$rc"
}

hf_snapshot_cached() {
  local REPO="$1"
  timeout --kill-after=5s "${HF_LOCAL_PROBE_TIMEOUT_SECONDS:-45}" python3 - "$REPO" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download(
    repo_id=sys.argv[1],
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
    local_files_only=True,
))

for path in root.rglob("*"):
    if path.is_symlink() and not path.exists():
        raise FileNotFoundError(f"broken symlink: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise RuntimeError(f"empty file: {path}")

for index_path in root.rglob("*.index.json"):
    payload = json.loads(index_path.read_text())
    for rel_name in set(payload.get("weight_map", {}).values()):
        shard_path = index_path.parent / rel_name
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing indexed shard: {shard_path}")
PY
}

hf_snapshot_dir_cached() {
  local REPO="$1"
  local TARGET="$2"
  timeout --kill-after=5s "${HF_LOCAL_PROBE_TIMEOUT_SECONDS:-45}" python3 - "$REPO" "$TARGET" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    local_files_only=True,
))

for path in root.rglob("*"):
    if path.is_symlink() and not path.exists():
        raise FileNotFoundError(f"broken symlink: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise RuntimeError(f"empty file: {path}")

for index_path in root.rglob("*.index.json"):
    payload = json.loads(index_path.read_text())
    for rel_name in set(payload.get("weight_map", {}).values()):
        shard_path = index_path.parent / rel_name
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing indexed shard: {shard_path}")
PY
}

# 0) Self-heal
sed -i 's/\r$//' "$0" 2>/dev/null || true

# 1) Sicherstellen, dass tools.config im Volume existiert
mkdir -p /workspace /workspace/LTX-2/outputs
mkdir -p \
  /workspace/agent_runs \
  /workspace/exports \
  /workspace/jobs \
  /workspace/status \
  /workspace/venvs

INIT_LOCK_FILE="/workspace/status/init.lock"
exec 9>"$INIT_LOCK_FILE"
if command_exists flock; then
  if ! flock -n 9; then
    log "[init] another init.sh is already running; refusing to start a second downloader."
    exit 0
  fi
  echo "$$" > /workspace/status/init.pid
fi
trap 'rm -f /workspace/status/init.pid' EXIT

if [ -f /app/tools.config ] && [ ! -f /workspace/tools.config ]; then
  cp -f /app/tools.config /workspace/tools.config
fi

sed -i 's/\r$//' /workspace/tools.config 2>/dev/null || true
source /workspace/tools.config 2>/dev/null || true
for config_file in \
  /workspace/config/director_llm.env \
  /workspace/config/director_llm.env.local
do
  if [ -f "$config_file" ]; then
    sed -i 's/\r$//' "$config_file" 2>/dev/null || true
    source "$config_file" 2>/dev/null || true
  fi
done

export PATH="/usr/local/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export GIT_TERMINAL_PROMPT=0
export GCM_INTERACTIVE=never
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PIP_NO_INPUT=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

log "[init] start pid=$$ cwd=$(pwd)"

# SoX systemweit sicherstellen
if ! command -v sox >/dev/null 2>&1; then
  log "[deps] installing sox"
  apt-get update
  apt-get install -y sox libsox-fmt-all
fi


# 2) Download-Funktionen
HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-3}"
HF_DOWNLOAD_TIMEOUT_SECONDS="${HF_DOWNLOAD_TIMEOUT_SECONDS:-3600}"
HF_DOWNLOAD_STALL_TIMEOUT_SECONDS="${HF_DOWNLOAD_STALL_TIMEOUT_SECONDS:-600}"
HF_DOWNLOAD_PROGRESS_SECONDS="${HF_DOWNLOAD_PROGRESS_SECONDS:-60}"
HF_DOWNLOAD_RETRY_SLEEP_SECONDS="${HF_DOWNLOAD_RETRY_SLEEP_SECONDS:-15}"
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"

function hf_download_all() {
  local REPO="$1"
  local TARGET="/workspace/.cache/hf/hub/models--${REPO//\//--}"
  local attempt
  if hf_snapshot_cached "$REPO"; then
    HF_DOWNLOAD_LAST_STATUS="existing"
    log "[hf] SKIP existing cached snapshot $REPO"
    return 0
  fi
  for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
    log "[hf] Downloading snapshot $REPO (attempt $attempt/$HF_DOWNLOAD_RETRIES) method=huggingface_hub.snapshot_download"
    if run_download_with_guard "snapshot $REPO" "$TARGET" python3 - "$REPO" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
    max_workers=int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "2")),
)
PY
    then
      return 0
    fi
    log "[hf] WARN: snapshot $REPO failed, timed out, or stalled."
    sleep "$HF_DOWNLOAD_RETRY_SLEEP_SECONDS"
  done
  log "[hf] ERROR: snapshot $REPO failed after $HF_DOWNLOAD_RETRIES attempts."
  return 1
}

function hf_download_file() {
  local REPO="$1"
  local FILE="$2"
  local TARGET="$3"
  local attempt
  if [ -s "$TARGET/$FILE" ]; then
    HF_DOWNLOAD_LAST_STATUS="existing"
    log "[hf] SKIP existing file $TARGET/$FILE"
    return 0
  fi
  for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
    log "[hf] Downloading $REPO/$FILE to $TARGET (attempt $attempt/$HF_DOWNLOAD_RETRIES) method=huggingface_hub.hf_hub_download"
    if run_download_with_guard "file $REPO/$FILE" "$TARGET/$FILE" python3 - "$REPO" "$FILE" "$TARGET" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])
PY
    then
      return 0
    fi
    log "[hf] WARN: file $REPO/$FILE failed, timed out, or stalled."
    sleep "$HF_DOWNLOAD_RETRY_SLEEP_SECONDS"
  done
  log "[hf] ERROR: file $REPO/$FILE failed after $HF_DOWNLOAD_RETRIES attempts."
  return 1
}

function hf_download_snapshot_to_dir() {
  local REPO="$1"
  local TARGET="$2"
  local attempt
  if hf_snapshot_dir_cached "$REPO" "$TARGET"; then
    HF_DOWNLOAD_LAST_STATUS="existing"
    log "[hf] SKIP existing snapshot $REPO in $TARGET"
    return 0
  fi
  for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
    log "[hf] Downloading snapshot $REPO to $TARGET (attempt $attempt/$HF_DOWNLOAD_RETRIES) method=huggingface_hub.snapshot_download"
    if run_download_with_guard "snapshot $REPO -> $TARGET" "$TARGET" python3 - "$REPO" "$TARGET" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    max_workers=int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "2")),
)
PY
    then
      return 0
    fi
    log "[hf] WARN: snapshot $REPO failed, timed out, or stalled."
    sleep "$HF_DOWNLOAD_RETRY_SLEEP_SECONDS"
  done
  log "[hf] ERROR: snapshot $REPO failed after $HF_DOWNLOAD_RETRIES attempts."
  return 1
}

# 3) Caches setzen
if [ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]; then
  if python3 -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  else
    export HF_HUB_ENABLE_HF_TRANSFER=0
  fi
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_DOWNLOAD_MAX_WORKERS
export HF_HOME=/workspace/.cache/hf
log "[hf] config HF_HOME=$HF_HOME hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER disable_xet=$HF_HUB_DISABLE_XET workers=$HF_DOWNLOAD_MAX_WORKERS"

MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"
QWEN_MODELS_DIR="/workspace/models/qwen3-tts"
QWEN_VENV="/workspace/venvs/qwen3-tts"
QWEN_PREBUILT_VENV="/opt/venvs/qwen3-tts"
QWEN_RUNTIME_FLAG="/workspace/status/qwen_tts_runtime_ready"
ACE_STEP_ROOT="/workspace/ACE-Step-1.5"
ACE_STEP_CKPT_DIR="$ACE_STEP_ROOT/checkpoints"
ACE_STEP_READY_FLAG="/workspace/status/ace_step_ready"
ACE_STEP_ENV_FLAG="/workspace/status/ace_step_env_ready"
DIRECTOR_LLM_MODEL_DIR="${DIRECTOR_LLM_MODEL_DIR:-/workspace/models/director/qwen3.6-35b-a3b/gguf}"
DIRECTOR_LLM_MODEL_FILE="${DIRECTOR_LLM_MODEL_FILE:-Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf}"
DIRECTOR_LLM_MODEL_PATH="${DIRECTOR_LLM_MODEL_PATH:-$DIRECTOR_LLM_MODEL_DIR/$DIRECTOR_LLM_MODEL_FILE}"
DIRECTOR_LLM_AUTO_SETUP="${DIRECTOR_LLM_AUTO_SETUP:-on}"
DIRECTOR_LLM_AUTO_START="${DIRECTOR_LLM_AUTO_START:-off}"
DIRECTOR_LLM_MODEL_READY_FLAG="/workspace/status/director_llm_model_ready"
DIRECTOR_LLM_SERVER_READY_FLAG="/workspace/status/director_llm_server_ready"
DIRECTOR_LLM_SETUP_FAILED_FLAG="/workspace/status/director_llm_setup_failed"
mkdir -p "$MODELS_DIR/ltx-2.3" "$MODELS_DIR/gemma-3" "$LORA_DIR"
mkdir -p "$QWEN_MODELS_DIR"
mkdir -p "$ACE_STEP_CKPT_DIR"
mkdir -p "$DIRECTOR_LLM_MODEL_DIR"
mkdir -p /workspace/scripts

if [ "${INIT_CHECK_ONLY:-0}" = "1" ]; then
  log "[init] CHECK_ONLY active; downloads and service starts will be skipped."
  log "[init] components: LTX-2=${DW_LTX2:-off} QwenTokenizer=${Qwen_TTS_Tokenizer:-off} QwenModel=${Qwen_TTS_Model:-off} Ace=${Ace_Step1_5:-off} ZTurbo=${Z_Image_Turbo:-off} ZBase=${Z_Image_Base:-off} DirectorSetup=$DIRECTOR_LLM_AUTO_SETUP DirectorStart=$DIRECTOR_LLM_AUTO_START"
  log "[init] paths: LTX=$MODELS_DIR Qwen=$QWEN_MODELS_DIR Ace=$ACE_STEP_CKPT_DIR Director=$DIRECTOR_LLM_MODEL_PATH"
  exit 0
fi

if [ "${INIT_DIRECTOR_ONLY:-0}" = "1" ]; then
  log "[init] DIRECTOR_ONLY active; running only Director download/startup path with downloads enabled."
  Qwen_TTS_Tokenizer=off
  Qwen_TTS_Model=off
  Ace_Step1_5=off
  Z_Image_Turbo=off
  Z_Image_Base=off
  DW_LTX2=off
  Lora1=off
  Lora2=off
  Lora3=off
  Lora4=off
  UPSCALER_INSTALL=off
  DIRECTOR_LLM_AUTO_SETUP=on
  DIRECTOR_LLM_AUTO_START=on
fi

# ----------------------------------------------------
# 4. Qwen3-TTS / Shared Runtime Sektion
# ----------------------------------------------------
if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ] || [ "${Ace_Step1_5:-off}" = "on" ]; then
  echo "[qwen] Preparing runtime environment..."
  mkdir -p "/workspace/venvs"
  if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
    rm -f "$QWEN_RUNTIME_FLAG"
  fi
  if [ "${Ace_Step1_5:-off}" = "on" ]; then
    rm -f "$ACE_STEP_ENV_FLAG"
  fi
  if [ ! -e "$QWEN_VENV" ] && [ -d "$QWEN_PREBUILT_VENV" ]; then
    ln -sfn "$QWEN_PREBUILT_VENV" "$QWEN_VENV"
  elif [ ! -f "$QWEN_VENV/bin/activate" ]; then
    python3 -m venv "$QWEN_VENV"
  fi

  if [ -f "$QWEN_VENV/bin/activate" ]; then
    if [ ! -d "$QWEN_PREBUILT_VENV" ] && \
       { [ "${Qwen_TTS_Tokenizer:-off}" != "on" ] && [ "${Qwen_TTS_Model:-off}" != "on" ] || [ ! -f "/workspace/status/qwen_tts_env_ready" ]; }; then
      "$QWEN_VENV/bin/pip" install --no-cache-dir -U pip setuptools wheel
      "$QWEN_VENV/bin/pip" install --no-cache-dir ninja packaging psutil pybind11
      "$QWEN_VENV/bin/pip" install --no-cache-dir \
        torch==2.7.0 \
        torchaudio==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128
      
      "$QWEN_VENV/bin/pip" install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation
      if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
        "$QWEN_VENV/bin/pip" install --no-cache-dir \
          "qwen-tts==0.1.1" \
          "transformers==4.57.3" \
          "accelerate==1.12.0"
      fi
    fi
    if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
      "$QWEN_VENV/bin/python" - <<'PY'
import qwen_tts
import transformers
print("qwen_tts ok", qwen_tts.__file__)
print("transformers", transformers.__version__)
PY
      touch "/workspace/status/qwen_tts_env_ready"
      touch "$QWEN_RUNTIME_FLAG"
    fi
  fi
fi

if [ "${Ace_Step1_5:-off}" = "on" ] && [ -f "$QWEN_VENV/bin/activate" ]; then
  echo "[ace-step] Preparing shared runtime in Qwen env..."

  if [ ! -f "$ACE_STEP_ENV_FLAG" ] && ! "$QWEN_VENV/bin/python" - <<'PY'
import transformers.configuration_utils as cu
import diffusers
import loguru
import toml
import modelscope
import torchvision
import torchao
assert hasattr(cu, "layer_type_validation")
PY
  then
    "$QWEN_VENV/bin/pip" install --no-cache-dir \
      "torchvision==0.22.0" \
      "diffusers" \
      "toml" \
      "loguru" \
      "modelscope" \
      "torchao" \
      "matplotlib>=3.7.5" \
      "scipy>=1.10.1" \
      "soundfile>=0.13.1" \
      "einops>=0.8.1" \
      "fastapi>=0.110.0" \
      "uvicorn[standard]>=0.27.0" \
      "numba>=0.63.1" \
      "vector-quantize-pytorch>=1.27.15" \
      "python-dotenv" \
      "xxhash"

    if [ -d "$ACE_STEP_ROOT/acestep/third_parts/nano-vllm" ]; then
      "$QWEN_VENV/bin/pip" install --no-cache-dir -e "$ACE_STEP_ROOT/acestep/third_parts/nano-vllm"
    fi
  fi

  "$QWEN_VENV/bin/python" - <<'PY'
import transformers
import transformers.configuration_utils as cu
import diffusers
import loguru
import toml
import modelscope
import torchvision
import torch

assert hasattr(cu, "layer_type_validation")
print("ace-step runtime ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("torchvision", torchvision.__version__)
PY
  touch "$ACE_STEP_ENV_FLAG"
fi

if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
  if [ -f "$QWEN_VENV/bin/activate" ]; then
    touch "/workspace/status/qwen_tts_env_ready"
    touch "$QWEN_RUNTIME_FLAG"
  fi
fi

if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ]; then
  echo "[qwen] Qwen_TTS_Tokenizer is ON. Checking tokenizer path..."
  if [ -d "/opt/models/qwen/Qwen3-TTS-Tokenizer-12Hz" ]; then
    ln -sfn "/opt/models/qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz"
  elif [ ! -f "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz/config.json" ]; then
    hf_download_snapshot_to_dir "Qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz"
  fi

  if [ -f "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz/config.json" ]; then
    touch "/workspace/status/qwen_tts_tokenizer_ready"
  fi
fi

if [ "${Qwen_TTS_Model:-off}" = "on" ]; then
  echo "[qwen] Qwen_TTS_Model is ON. Checking model path..."
  if [ -d "/opt/models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" ]; then
    ln -sfn "/opt/models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  elif [ ! -f "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json" ]; then
    hf_download_snapshot_to_dir "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  fi

  if [ -f "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json" ]; then
    touch "/workspace/status/qwen_tts_model_ready"
  fi
fi

# ----------------------------------------------------
# 5. ACE-Step 1.5 Sektion
# ----------------------------------------------------
if [ "${Ace_Step1_5:-off}" = "on" ]; then
  echo "[ace-step] Ace_Step1_5 is ON. Checking model path..."
  if [ -f "$ACE_STEP_CKPT_DIR/acestep-v15-turbo/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/vae/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/Qwen3-Embedding-0.6B/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/acestep-5Hz-lm-1.7B/config.json" ]; then
    echo "[ace-step] Main model already present."
  else
    hf_download_snapshot_to_dir "ACE-Step/Ace-Step1.5" "$ACE_STEP_CKPT_DIR"
  fi

  if [ -f "$ACE_STEP_CKPT_DIR/acestep-v15-turbo/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/vae/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/Qwen3-Embedding-0.6B/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/acestep-5Hz-lm-1.7B/config.json" ]; then
    touch "$ACE_STEP_READY_FLAG"
  fi
fi

# ----------------------------------------------------
# 6. Z-Image Sektion
# ----------------------------------------------------
if [ "${Z_Image_Turbo:-off}" = "on" ]; then
  echo "[zimage] Z_Image_Turbo is ON. Checking cache..."
  hf_download_all "Tongyi-MAI/Z-Image-Turbo"
  if [ "${HF_DOWNLOAD_LAST_STATUS:-}" != "skipped" ]; then
    touch "/workspace/status/zimage_ready"
  else
    log "[zimage] SKIP mode active; not marking zimage_ready."
  fi
fi

if [ "${Z_Image_Base:-off}" = "on" ]; then
  echo "[zimage] Z_Image_Base is ON. Starting download..."
  hf_download_all "Tongyi-MAI/Z-Image"
  if [ "${HF_DOWNLOAD_LAST_STATUS:-}" != "skipped" ]; then
    touch "/workspace/status/zimage_base_ready"
  else
    log "[zimage] SKIP mode active; not marking zimage_base_ready."
  fi
fi

# ----------------------------------------------------
# 7. LTX-2 Basis-Modelle
# ----------------------------------------------------
if [ "${DW_LTX2:-off}" = "on" ] && \
   { [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-dev.safetensors" ] || \
   [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" ] || \
   [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors" ] || \
   [ ! -f "$MODELS_DIR/gemma-3/config.json" ]; }; then
  echo "🚀 Hauptmodelle fehlen – Starte Setup..."

  if [ -n "${HF_TOKEN:-}" ]; then
    timeout --kill-after=10s 60s python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)" || \
      log "[hf] WARN: token login failed or timed out; continuing with HF_TOKEN environment."
  fi

  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-22b-dev.safetensors" "$MODELS_DIR/ltx-2.3"
  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2.3"
  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2.3"

  echo "🚀 Lade Gemma-3..."
  hf_download_snapshot_to_dir "google/gemma-3-12b-it-qat-q4_0-unquantized" "$MODELS_DIR/gemma-3"
fi

# ----------------------------------------------------
# 8. Director-LLM Sektion
# ----------------------------------------------------
if [ "$DIRECTOR_LLM_AUTO_SETUP" = "on" ]; then
  echo "[director-llm] Ensuring local Qwen3.6 Director model..."
  rm -f "$DIRECTOR_LLM_MODEL_READY_FLAG"
  rm -f "$DIRECTOR_LLM_SERVER_READY_FLAG"
  rm -f "$DIRECTOR_LLM_SETUP_FAILED_FLAG"

  if [ -f "/workspace/scripts/download_director_model.py" ]; then
    director_download_ok=0
    for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
      log "[director-llm] Download attempt $attempt/$HF_DOWNLOAD_RETRIES target=$DIRECTOR_LLM_MODEL_PATH"
      if [ -s "$DIRECTOR_LLM_MODEL_PATH" ]; then
        log "[director-llm] SKIP existing model $DIRECTOR_LLM_MODEL_PATH"
        director_download_ok=1
        break
      fi
      if run_download_with_guard "director model $DIRECTOR_LLM_MODEL_FILE" "$DIRECTOR_LLM_MODEL_PATH" python3 /workspace/scripts/download_director_model.py; then
        director_download_ok=1
        break
      fi
      log "[director-llm] WARN: model download failed, timed out, or stalled."
      sleep "$HF_DOWNLOAD_RETRY_SLEEP_SECONDS"
    done

    if [ "$director_download_ok" = "1" ]; then
      if [ -f "$DIRECTOR_LLM_MODEL_PATH" ]; then
        touch "$DIRECTOR_LLM_MODEL_READY_FLAG"
        echo "[director-llm] Model ready at $DIRECTOR_LLM_MODEL_PATH"
      else
        echo "[director-llm] ERROR: download script completed without expected model file $DIRECTOR_LLM_MODEL_PATH"
        touch "$DIRECTOR_LLM_SETUP_FAILED_FLAG"
      fi
    else
      echo "[director-llm] ERROR: model download/preparation failed."
      touch "$DIRECTOR_LLM_SETUP_FAILED_FLAG"
    fi
  else
    echo "[director-llm] WARN: /workspace/scripts/download_director_model.py missing; skipping Director model setup."
    touch "$DIRECTOR_LLM_SETUP_FAILED_FLAG"
  fi

  if [ "$DIRECTOR_LLM_AUTO_START" = "on" ] && [ -f "$DIRECTOR_LLM_MODEL_PATH" ]; then
    if [ -f "/workspace/scripts/serve_director_llm.sh" ]; then
      chmod +x "/workspace/scripts/serve_director_llm.sh" 2>/dev/null || true
      DIRECTOR_LLM_START_TIMEOUT_SECONDS="${DIRECTOR_LLM_START_TIMEOUT_SECONDS:-1800}"
      log "[director-llm] Starting local Director server timeout=${DIRECTOR_LLM_START_TIMEOUT_SECONDS}s"
      if timeout --kill-after=30s "$DIRECTOR_LLM_START_TIMEOUT_SECONDS" env DIRECTOR_LLM_DAEMON=1 bash /workspace/scripts/serve_director_llm.sh; then
        touch "$DIRECTOR_LLM_SERVER_READY_FLAG"
        echo "[director-llm] Local Director server is ready."
      else
        echo "[director-llm] ERROR: local Director server failed to start."
        touch "$DIRECTOR_LLM_SETUP_FAILED_FLAG"
      fi
    else
      echo "[director-llm] WARN: /workspace/scripts/serve_director_llm.sh missing or not executable; skipping auto-start."
      touch "$DIRECTOR_LLM_SETUP_FAILED_FLAG"
    fi
  fi
fi

# ----------------------------------------------------
# 9. LoRA Sektion
# ----------------------------------------------------
echo "📥 Prüfe LoRA Downloads..."

[[ "${Lora1:-off}" == "on" && ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]] && \
  hf_download_file "kabachuha/ltx2-cakeify" "ltx2-cakeify-v2.safetensors" "$LORA_DIR"

[[ "${Lora2:-off}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]] && \
  hf_download_file "Lightricks/LTX-2-19b-IC-LoRA-Detailer" "ltx-2-19b-ic-lora-detailer.safetensors" "$LORA_DIR"

[[ "${Lora3:-off}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]] && \
  hf_download_file "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" "ltx-2-19b-lora-camera-control-static.safetensors" "$LORA_DIR"

[[ "${Lora4:-off}" == "on" && ! -f "$LORA_DIR/LTX-2-Image2Vid-Adapter.safetensors" ]] && \
  hf_download_file "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa" "LTX-2-Image2Vid-Adapter.safetensors" "$LORA_DIR"

# ----------------------------------------------------
# 10. Abschluss
# ----------------------------------------------------
chmod -R 777 "$MODELS_DIR" || true
chmod -R 777 "$QWEN_MODELS_DIR" || true
chmod -R 777 "$QWEN_VENV" || true
chmod -R 777 "$ACE_STEP_CKPT_DIR" || true
chmod -R 777 "/workspace/models/director" || true
chmod +x /workspace/scripts/*.sh 2>/dev/null || true
chmod +x /workspace/scripts/*.py 2>/dev/null || true
echo "🏁 init.sh erfolgreich beendet."
if [ "${INIT_SKIP_DOWNLOADS:-0}" = "1" ] || [ "${INIT_DIRECTOR_ONLY:-0}" = "1" ]; then
  log "[init] diagnostic/partial init mode active; not marking init_done."
else
  touch /workspace/status/init_done
fi

# ----------------------------------------------------
# 11. Optional: Real-ESRGAN AI Installer (nach init_done)
# ----------------------------------------------------
if [ "${UPSCALER_INSTALL:-off}" = "on" ]; then
  UPSCALER_INSTALLER="/workspace/upscaler_installer_minimal/install_realesrgan_ai_pod.sh"
  if [ -f "$UPSCALER_INSTALLER" ]; then
    chmod +x "$UPSCALER_INSTALLER" 2>/dev/null || true
    echo "🛠️ Starte optionalen Upscaler-Installer..."
    if bash "$UPSCALER_INSTALLER"; then
      touch /workspace/status/upscaler_install_done
      echo "✅ Upscaler-Installer erfolgreich."
    else
      touch /workspace/status/upscaler_install_failed
      echo "❌ Upscaler-Installer fehlgeschlagen."
    fi
  else
    echo "⚠️ Upscaler-Installer nicht gefunden: $UPSCALER_INSTALLER"
 



 fi
fi
