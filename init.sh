#!/bin/bash
set -e

# 1. Config laden
source /workspace/tools.config 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

MODELS_DIR="/workspace/checkpoints"
mkdir -p "$MODELS_DIR/ltx-2"
mkdir -p "$MODELS_DIR/gemma-3"

# 2. Modell-Download nur wenn DW_LTX2=on
if [ "${DW_LTX2:-off}" = "on" ]; then
    echo "📥 DW_LTX2 ist ON – Starte Downloads..."
    
    # Login für Gemma 3
    if [ -n "${HF_TOKEN:-}" ]; then
        huggingface-cli login --token "$HF_TOKEN"
    fi

    # LTX-2 & Gemma Downloads
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp8.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download google/gemma-3-12b-it --local-dir "$MODELS_DIR/gemma-3" --local-dir-use-symlinks False

    echo "✅ Downloads abgeschlossen."
else
    echo "⏭️  DW_LTX2 ist OFF – Überspringe Modell-Downloads."
fi

# 3. Allgemeine Aufgaben (laufen immer, wenn init.sh startet)
echo "🧹 JUPYTER-FIX: Entferne Cache-Leichen..."
find "$MODELS_DIR" -name ".cache" -type d -exec rm -rf {} +

# Hier kannst du in Zukunft einfach neue Befehle hinzufügen!
echo "🏁 init.sh fertig."
touch /workspace/status/init_done