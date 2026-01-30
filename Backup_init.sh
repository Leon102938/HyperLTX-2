#!/bin/bash
set -e

# 1. Pfad & Tools sicherstellen
export PATH="$PATH:/root/.local/bin"
mkdir -p /workspace/status

# Installation (hf_transfer für max Speed)
python3 -m pip install --upgrade huggingface_hub hf_transfer

# 2. Config laden & Bindestrich-Fix
if [ -f /workspace/tools.config ]; then
    eval $(sed 's/-/_/g' /workspace/tools.config)
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

# Feste Pfad-Definition
MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"
mkdir -p "$MODELS_DIR/ltx-2" "$MODELS_DIR/gemma-3" "$LORA_DIR"

# Funktion für den Download über Python-API (Schusssicher!)
python_download() {
    # $1 = Repo ID, $2 = Filename (optional), $3 = Local Dir (optional)
    if [ -z "$2" ]; then
        # Snapshot download für ganze Repos (Z-Image)
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$1', ignore_patterns=['assets/*', 'README.md'])"
    else
        # Single file download für LTX-2 und LoRAs
        python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$1', filename='$2', local_dir='$3', local_dir_use_symlinks=False)"
    fi
}

# --- Z-Image Sektion ---
if [ "$Z_Image_Turbo" = "on" ]; then
    echo "[zimage] Z_Image_Turbo is ON..."
    python_download "Tongyi-MAI/Z-Image-Turbo"
    touch "/workspace/status/zimage_turbo_ready"
fi

if [ "$Z_Image_Base" = "on" ]; then
    echo "[zimage] Z_Image_Base is ON..."
    python_download "Tongyi-MAI/Z-Image"
    touch "/workspace/status/zimage_base_ready"
fi

# --- 2. Hauptmodelle (LTX-2 & Gemma) ---
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "🚀 Hauptmodelle fehlen – Starte Setup..."
    python_download "Lightricks/LTX-2" "ltx-2-19b-dev-fp8.safetensors" "$MODELS_DIR/ltx-2"
    python_download "Lightricks/LTX-2" "ltx-2-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2"
    python_download "Lightricks/LTX-2" "ltx-2-19b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2"
    # Gemma 3 (ganzes Repo)
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/gemma-3-12b-it', local_dir='$MODELS_DIR/gemma-3', local_dir_use_symlinks=False)"
fi

# --- 3. LoRA Sektion ---
echo "📥 Prüfe LoRA Downloads..."

if [ "$Lora1" = "on" ] && [ ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]; then
    python_download "kabachuha/ltx2-cakeify" "ltx2-cakeify-v2.safetensors" "$LORA_DIR"
fi

if [ "$Lora2" = "on" ] && [ ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]; then
    python_download "Lightricks/LTX-2-19b-IC-LoRA-Detailer" "ltx-2-19b-ic-lora-detailer.safetensors" "$LORA_DIR"
fi

if [ "$Lora3" = "on" ] && [ ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]; then
    python_download "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" "ltx-2-19b-lora-camera-control-static.safetensors" "$LORA_DIR"
fi

if [ "$Lora4" = "on" ] && [ ! -f "$LORA_DIR/ltx2_i2v_adapter.safetensors" ]; then
    python_download "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa" "ltx2_i2v_adapter.safetensors" "$LORA_DIR"
fi

# 4. Abschluss
chmod -R 777 "$MODELS_DIR"
echo "✅ Alle Downloads abgeschlossen."
touch /workspace/status/init_done