#!/bin/bash
set -euo pipefail

# 0) Self-heal
sed -i 's/\r$//' "$0" 2>/dev/null || true

# 1) Sicherstellen, dass tools.config im Volume existiert
mkdir -p /workspace /workspace/LTX-2/outputs
mkdir -p /workspace /workspace/status
if [ -f /app/tools.config ] && [ ! -f /workspace/tools.config ]; then
  cp -f /app/tools.config /workspace/tools.config
fi

sed -i 's/\r$//' /workspace/tools.config 2>/dev/null || true
source /workspace/tools.config 2>/dev/null || true

export PATH="/usr/local/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH"

# 2) Download-Funktionen
function hf_download_all() {
  local REPO="$1"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors","*.json","*.txt"]
)
PY
}

function hf_download_file() {
  local REPO="$1"
  local FILE="$2"
  local TARGET="$3"
  python3 - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$TARGET", local_dir_use_symlinks=False)
PY
}

function hf_download_snapshot_to_dir() {
  local REPO="$1"
  local TARGET="$2"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    local_dir="$TARGET",
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
}

# 3) Caches setzen
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"
QWEN_MODELS_DIR="/workspace/models/qwen3-tts"
QWEN_VENV="/workspace/venvs/qwen3-tts"
QWEN_PREBUILT_VENV="/opt/venvs/qwen3-tts"
QWEN_RUNTIME_FLAG="/workspace/status/qwen_tts_runtime_ready"
mkdir -p "$MODELS_DIR/ltx-2.3" "$MODELS_DIR/gemma-3" "$LORA_DIR"
mkdir -p "$QWEN_MODELS_DIR"

# ----------------------------------------------------
# 4. Qwen3-TTS Sektion
# ----------------------------------------------------
if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
  echo "[qwen] Preparing runtime environment..."
  mkdir -p "/workspace/venvs"
  rm -f "$QWEN_RUNTIME_FLAG"
  if [ ! -e "$QWEN_VENV" ] && [ -d "$QWEN_PREBUILT_VENV" ]; then
    ln -sfn "$QWEN_PREBUILT_VENV" "$QWEN_VENV"
  elif [ ! -f "$QWEN_VENV/bin/activate" ]; then
    python3 -m venv "$QWEN_VENV"
  fi

  if [ -f "$QWEN_VENV/bin/activate" ]; then
    if [ ! -d "$QWEN_PREBUILT_VENV" ]; then
      "$QWEN_VENV/bin/pip" install --no-cache-dir -U pip setuptools wheel
      "$QWEN_VENV/bin/pip" install --no-cache-dir ninja packaging psutil pybind11
      "$QWEN_VENV/bin/pip" install --no-cache-dir \
        torch==2.7.0 \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
      "$QWEN_VENV/bin/pip" install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation
      "$QWEN_VENV/bin/pip" install --no-cache-dir \
        "qwen-tts==0.1.1" \
        "transformers==4.57.3" \
        "accelerate==1.12.0"
    fi
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
# 5. Z-Image Sektion
# ----------------------------------------------------
if [ "${Z_Image_Turbo:-off}" = "on" ]; then
  echo "[zimage] Z_Image_Turbo is ON. Checking cache..."
  hf_download_all "Tongyi-MAI/Z-Image-Turbo"
  touch "/workspace/status/zimage_ready"
fi

if [ "${Z_Image_Base:-off}" = "on" ]; then
  echo "[zimage] Z_Image_Base is ON. Starting download..."
  hf_download_all "Tongyi-MAI/Z-Image"
  touch "/workspace/status/zimage_base_ready"
fi

# ----------------------------------------------------
# 6. LTX-2 Basis-Modelle
# ----------------------------------------------------
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
  echo "🚀 Hauptmodelle fehlen – Starte Setup..."

  if [ -n "${HF_TOKEN:-}" ]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
  fi

  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-22b-dev.safetensors" "$MODELS_DIR/ltx-2.3"
  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2.3"
  hf_download_file "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2.3"

echo "🚀 Lade Gemma-3..."
python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
    local_dir="$MODELS_DIR/gemma-3",
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
fi

# ----------------------------------------------------
# 7. LoRA Sektion
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
# 8. Abschluss
# ----------------------------------------------------
chmod -R 777 "$MODELS_DIR" || true
chmod -R 777 "$QWEN_MODELS_DIR" || true
chmod -R 777 "$QWEN_VENV" || true
echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done

# ----------------------------------------------------
# 9. Optional: Real-ESRGAN AI Installer (nach init_done)
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
