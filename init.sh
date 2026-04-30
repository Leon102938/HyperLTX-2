#!/bin/bash
set -euo pipefail

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
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "[init] another init.sh is already running; exiting."
    exit 0
  fi
fi

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
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
export PIP_NO_INPUT=1


# SoX systemweit sicherstellen
if ! command -v sox >/dev/null 2>&1; then
  apt-get update
  apt-get install -y sox libsox-fmt-all
fi


# 2) Download-Funktionen
function hf_download_all() {
  local REPO="$1"
  echo "[hf] snapshot repo=$REPO hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  if python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors","*.json","*.txt"]
)
PY
  then
    echo "[hf] done repo=$REPO"
    return 0
  fi

  echo "[hf] fail repo=$REPO"
  return 1
}

function hf_download_file() {
  local REPO="$1"
  local FILE="$2"
  local TARGET="$3"
  if [ -s "$TARGET/$FILE" ]; then
    echo "[hf] skip existing file=$TARGET/$FILE"
    return 0
  fi

  echo "[hf] file repo=$REPO file=$FILE target=$TARGET hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  if python3 - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$TARGET", local_dir_use_symlinks=False, resume_download=True)
PY
  then
    echo "[hf] done file=$TARGET/$FILE"
    return 0
  fi

  echo "[hf] fail file=$REPO/$FILE"
  return 1
}

function hf_download_snapshot_to_dir() {
  local REPO="$1"
  local TARGET="$2"
  if [ -f "$TARGET/config.json" ] && ! find "$TARGET" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
    echo "[hf] skip existing snapshot target=$TARGET"
    return 0
  fi

  echo "[hf] snapshot repo=$REPO target=$TARGET hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  if python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    local_dir="$TARGET",
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
  then
    echo "[hf] done target=$TARGET"
    return 0
  fi

  echo "[hf] fail repo=$REPO target=$TARGET"
  return 1
}

function gemma_model_ready() {
  local TARGET="$1"
  [ -f "$TARGET/config.json" ] || return 1
  [ -f "$TARGET/tokenizer.model" ] || return 1
  [ -f "$TARGET/tokenizer.json" ] || return 1
  [ -f "$TARGET/tokenizer_config.json" ] || return 1
  [ -f "$TARGET/preprocessor_config.json" ] || return 1
  [ -f "$TARGET/model.safetensors.index.json" ] || return 1
  ! find "$TARGET" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . || return 1
  python3 - "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
for shard in set(index.get("weight_map", {}).values()):
    path = root / shard
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(1)
for path in root.rglob("*"):
    if path.is_file() and path.stat().st_size == 0:
        raise SystemExit(1)
PY
}

# 3) Caches setzen
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME=/workspace/.cache/hf

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
Qwen3_VL_Review="${Qwen3_VL_Review:-off}"
Vision_Review_Model="${Vision_Review_Model:-off}"
QWEN3_VL_MODEL_DIR="/workspace/models/Qwen3-VL-4B-Instruct-FP8"
QWEN3_VL_READY_FLAG="/workspace/status/qwen3_vl_ready"
mkdir -p "$MODELS_DIR/ltx-2.3" "$MODELS_DIR/gemma-3" "$LORA_DIR"
mkdir -p "$QWEN_MODELS_DIR"
mkdir -p "$ACE_STEP_CKPT_DIR"
mkdir -p "$DIRECTOR_LLM_MODEL_DIR"
mkdir -p /workspace/scripts

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
  if hf_download_all "Tongyi-MAI/Z-Image-Turbo"; then
    touch "/workspace/status/zimage_ready"
  fi
fi

if [ "${Z_Image_Base:-off}" = "on" ]; then
  echo "[zimage] Z_Image_Base is ON. Starting download..."
  if hf_download_all "Tongyi-MAI/Z-Image"; then
    touch "/workspace/status/zimage_base_ready"
  fi
fi

# ----------------------------------------------------
# 7. LTX-2 Basis-Modelle
# ----------------------------------------------------
if [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-dev.safetensors" ] || \
   [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" ] || \
   [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors" ] || \
   ! gemma_model_ready "$MODELS_DIR/gemma-3"; then
  echo "🚀 Hauptmodelle fehlen – Starte Setup..."

  if [ -n "${HF_TOKEN:-}" ]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
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
    if python3 /workspace/scripts/download_director_model.py; then
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
      if DIRECTOR_LLM_DAEMON=1 bash /workspace/scripts/serve_director_llm.sh; then
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
# 8b. Optional Qwen3-VL Vision Review Modell
# ----------------------------------------------------
if [ "$Qwen3_VL_Review" = "on" ] || [ "$Vision_Review_Model" = "on" ]; then
  echo "[qwen3-vl] Qwen3-VL review model is ON. Checking model path..."
  rm -f "$QWEN3_VL_READY_FLAG"
  if python3 /workspace/scripts/download_qwen3_vl_model.py; then
    touch "$QWEN3_VL_READY_FLAG"
    echo "[qwen3-vl] Model ready at $QWEN3_VL_MODEL_DIR"
  else
    rm -f "$QWEN3_VL_READY_FLAG"
    echo "[qwen3-vl] ERROR: model download/verify failed."
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
touch /workspace/status/init_done

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
