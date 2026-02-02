#!/bin/bash
set -e

# 1. Selbstheilung & Pfade
sed -i 's/\r$//' "$0" 2>/dev/null || true
sed -i 's/\r$//' /workspace/tools.config 2>/dev/null || true
export PATH="/usr/local/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH"

# 2. Download-Funktionen (Die neuen "Fenster"-Methoden)
function hf_download_all() {
    local REPO=$1
    shift
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$REPO", local_dir_use_symlinks=False, resume_download=True, $@)
PY
}

function hf_download_file() {
    local REPO=$1
    local FILE=$2
    local TARGET=$3
    python3 - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$TARGET", local_dir_use_symlinks=False)
PY
}

# 3. Config laden & Caches setzen
source /workspace/tools.config 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"

mkdir -p /workspace/status "$MODELS_DIR/ltx-2" "$MODELS_DIR/gemma-3" "$LORA_DIR"

# ----------------------------------------------------
# 4. Z-Image Sektion
# ----------------------------------------------------
if [ "${Z_Image_Turbo}" = "on" ]; then
    echo "[zimage] Z_Image_Turbo is ON. Checking cache..."
    hf_download_all "Tongyi-MAI/Z-Image-Turbo" "allow_patterns=['*.safetensors', '*.json', '*.txt']"
    touch "/workspace/status/zimage_turbo_ready"
fi

if [ "${Z_Image_Base}" = "on" ]; then
    echo "[zimage] Z_Image_Base is ON. Starting download..."
    hf_download_all "Tongyi-MAI/Z-Image" "allow_patterns=['*.safetensors', '*.json', '*.txt']"
    touch "/workspace/status/zimage_base_ready"
fi

# ----------------------------------------------------
# 5. LTX-2 Basis-Modelle
# ----------------------------------------------------
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "🚀 Hauptmodelle fehlen – Starte Setup..."
    
    # Login nur wenn Token da ist
    if [ -n "$HF_TOKEN" ]; then
        python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    fi

    hf_download_file "Lightricks/LTX-2" "ltx-2-19b-dev-fp8.safetensors" "$MODELS_DIR/ltx-2"
    hf_download_file "Lightricks/LTX-2" "ltx-2-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2"
    hf_download_file "Lightricks/LTX-2" "ltx-2-19b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2"
    
    echo "🚀 Lade Gemma-3..."
    hf_download_all "google/gemma-3-12b-it" "local_dir='$MODELS_DIR/gemma-3'"
fi

# ----------------------------------------------------
# 6. LoRA Sektion
# ----------------------------------------------------
echo "📥 Prüfe LoRA Downloads..."

# Lora 1: Cakeify
[[ "${Lora1}" == "on" && ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]] && \
    hf_download_file "kabachuha/ltx2-cakeify" "ltx2-cakeify-v2.safetensors" "$LORA_DIR"

# Lora 2: Detailer
[[ "${Lora2}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]] && \
    hf_download_file "Lightricks/LTX-2-19b-IC-LoRA-Detailer" "ltx-2-19b-ic-lora-detailer.safetensors" "$LORA_DIR"

# Lora 3: Static Camera
[[ "${Lora3}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]] && \
    hf_download_file "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" "ltx-2-19b-lora-camera-control-static.safetensors" "$LORA_DIR"

# Lora 4: Image2Video Adapter
[[ "${Lora4}" == "on" && ! -f "$LORA_DIR/LTX-2-Image2Vid-Adapter.safetensors" ]] && \
    hf_download_file "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa" "LTX-2-Image2Vid-Adapter.safetensors" "$LORA_DIR"

# ----------------------------------------------------
# 7. Abschluss
# ----------------------------------------------------
chmod -R 777 "$MODELS_DIR"
echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done