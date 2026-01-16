#!/bin/bash
set -e


mkdir -p /workspace/status

# 1. Config laden & Caches setzen
source /workspace/tools.config 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/hf

# Feste Pfad-Definition (Passend zu deinem ti2vid_two_stages Skript)
MODELS_DIR="/workspace/LTX-2/checkpoints"
mkdir -p "$MODELS_DIR/ltx-2"
mkdir -p "$MODELS_DIR/gemma-3"

# 2. Intelligenter Auto-Download
# Prüft, ob das Hauptmodell fehlt. Wenn ja -> Startet Download.
if [ ! -f "$MODELS_DIR/ltx-2/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "🚀 Modelle fehlen – Starte automatischen Setup-Prozess..."
    
    # Login nur wenn Token vorhanden (wichtig für Gemma 3)
    if [ -n "$HF_TOKEN" ]; then
        echo "🔑 HF_TOKEN gefunden. Logge ein..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "⚠️  Kein HF_TOKEN in RunPod gesetzt! Gemma-3 Download könnte scheitern."
    fi

    # LTX-2 & Gemma Downloads in die feste Struktur
    echo "📥 Lade Modelle nach $MODELS_DIR..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp8.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR/ltx-2" --local-dir-use-symlinks False
    huggingface-cli download google/gemma-3-12b-it --local-dir "$MODELS_DIR/gemma-3" --local-dir-use-symlinks False

    echo "✅ Alle Downloads abgeschlossen."
else
    echo "✅ Modelle bereits in $MODELS_DIR vorhanden. Überspringe Download."
fi

# 3. Rechte-Fix für das RunPod-Interface (Damit Ordner immer öffenbar sind)
chmod -R 777 "$MODELS_DIR"

echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done