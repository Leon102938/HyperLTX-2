#!/bin/bash

set -e

mkdir -p /workspace/status

# 1. Config laden & Caches setzen
source /workspace/tools.config 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

# Feste Pfad-Definition
MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"
mkdir -p "$MODELS_DIR/ltx-2"
mkdir -p "$MODELS_DIR/gemma-3"
mkdir -p "$LORA_DIR"






# Z-Image Turbo
# Wir prüfen, ob der Wert in der Config auf "on" steht
if [ "$Z-Image-Turbo" = "on" ] || [ "$Z_Image_Turbo" = "on" ]; then
    echo "[zimage] Z_Image_Turbo is ON. Checking cache..."
    huggingface-cli download Tongyi-MAI/Z-Image-Turbo --exclude "assets/*" "README.md"
    touch "/workspace/status/zimage_turbo_ready"
    echo "[zimage] Turbo ready."
else
    echo "[zimage] Turbo is OFF. Skipping."
fi

# Z-Image Base
if [ "$Z-Image-Base" = "on" ] || [ "$Z_Image_Base" = "on" ]; then
    echo "[zimage] Z_Image_Base is ON. Starting 20GB download..."
    huggingface-cli download Tongyi-MAI/Z-Image --exclude "assets/*" "README.md"
    touch "/workspace/status/zimage_base_ready"
    echo "[zimage] Base ready."
else
    echo "[zimage] Base is OFF. Skipping."
fi








# 2. Intelligenter Auto-Download Basis-Modelle
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "🚀 Hauptmodelle fehlen – Starte Setup..."


    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    fi

    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp8.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download google/gemma-3-12b-it --local-dir "$MODELS_DIR/gemma-3" --local-dir-use-symlinks False
fi

# --- 3. LoRA Sektion (Einzeln schaltbar) ---
echo "📥 Prüfe LoRA Downloads..."

# LoRA 1: Cakeify
if [ "$Lora1" = "on" ] && [ ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]; then
    echo "Lade Lora 1 (Cakeify)..."
    huggingface-cli download kabachuha/ltx2-cakeify ltx2-cakeify-v2.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

# LoRA 2: Detailer
if [ "$Lora2" = "on" ] && [ ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]; then
    echo "Lade Lora 2 (Detailer)..."
    huggingface-cli download Lightricks/LTX-2-19b-IC-LoRA-Detailer ltx-2-19b-ic-lora-detailer.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

# LoRA 3: Static Camera
if [ "$Lora3" = "on" ] && [ ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]; then
    echo "Lade Lora 3 (Static Camera)..."
    huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Static ltx-2-19b-lora-camera-control-static.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

# LoRA 4: Image2Video Adapter (Neu!)
if [ "$Lora4" = "on" ] && [ ! -f "$LORA_DIR/ltx2_i2v_adapter.safetensors" ]; then
    echo "Lade Lora 4 (I2V Adapter)..."
    huggingface-cli download MachineDelusions/LTX-2_Image2Video_Adapter_LoRa ltx2_i2v_adapter.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi




echo "✅ Alle Downloads (inkl. spezieller LoRAs) abgeschlossen."

# 4. Rechte-Fix
chmod -R 777 "$MODELS_DIR"

echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done