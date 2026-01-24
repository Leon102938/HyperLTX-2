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

# 2. Intelligenter Auto-Download
# Prüft, ob das Hauptmodell fehlt. Wenn ja -> Startet Download.
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "🚀 Hauptmodelle fehlen – Starte automatischen Setup-Prozess..."
    
    # Login nur wenn Token vorhanden
    if [ -n "$HF_TOKEN" ]; then
        echo "🔑 HF_TOKEN gefunden. Logge ein..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "⚠️  Kein HF_TOKEN in RunPod gesetzt! Gemma-3 Download könnte scheitern."
    fi

    # LTX-2 & Gemma Downloads
    echo "📥 Lade Basis-Modelle nach $MODELS_DIR..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp8.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download google/gemma-3-12b-it --local-dir "$MODELS_DIR/gemma-3" --local-dir-use-symlinks False
fi

# 3. LoRA Downloads (Zusätzlich hinzugefügt)
echo "📥 Prüfe LoRAs in $LORA_DIR..."

# Cakeify V2
if [ ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]; then
    huggingface-cli download Lightricks/LTX-2-Cakeify-V2 ltx2-cakeify-v2.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

# Detailer LoRA
if [ ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]; then
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-ic-lora-detailer.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

# Static Camera Control
if [ ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]; then
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-lora-camera-control-static.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False
fi

echo "✅ Alle Downloads (inkl. LoRAs) abgeschlossen."

# 4. Rechte-Fix für das RunPod-Interface
chmod -R 777 "$MODELS_DIR"

echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done