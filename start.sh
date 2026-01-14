#!/bin/bash

set -euo pipefail


# ============ 📌 FIX 1: PFAD & VERSION PINNEN ============
PROJECT_ROOT="/workspace/LTX-2"
cd "$PROJECT_ROOT"

# Berechtigungen pinnen (Löst dein Problem mit dem Checkpoints-Ordner)
chmod -R 777 "$PROJECT_ROOT" || true

# ============ 🐍 FIX 2: PYTHONPATH ============
# Stellt sicher, dass ltx_core gefunden wird
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/packages/ltx-core/src:$PROJECT_ROOT/packages/ltx-pipelines/src:${PYTHONPATH:-}"

# ============ 🛠️ FIX 3: GEMMA 3 SUPPORT ============
# Erzwingt das Update in der richtigen Python-Umgebung
echo "🛠️ Update Transformers für Gemma 3..."
python3.13 -m pip install --upgrade transformers accelerate

# ============ 🔧 PyTorch & Hardware Specs ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
if [ -w /etc/profile.d ]; then
  echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" > /etc/profile.d/pytorch_alloc.sh || true
fi

[ -f "./tools.config" ] && source ./tools.config


# 🌍 BASE_URL automatisch setzen (RUNPOD_POD_ID sicher expandieren)
echo "🌐 Ermittle dynamische RunPod Proxy-URL..."
POD_ID="${RUNPOD_POD_ID:-}"
if [ -z "$POD_ID" ]; then
  echo "❌ FEHLER: RUNPOD_POD_ID nicht gesetzt – .env nicht geschrieben!"
else
  BASE_URL="https://${POD_ID}-8000.proxy.runpod.net"
  export BASE_URL
  echo "BASE_URL=$BASE_URL" > /workspace/.env
  echo "✅ BASE_URL erfolgreich gesetzt: $BASE_URL"
fi


# ============ 🔷 JUPYTERLAB THEME ============
mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
echo '{ "theme": "JupyterLab Dark" }' \
  > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# ============ 🔷 JUPYTERLAB (Port 8888) ============
if [ "${JUPYTER:-off}" = "on" ]; then
  echo "🧠 Starte JupyterLab (Port 8888)..."
  nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.disable_check_xsrf=True \
    --NotebookApp.notebook_dir='/workspace' \
    --ServerApp.allow_origin='*' \
    > /workspace/jupyter.log 2>&1 &
fi

# ============ 🔷 FASTAPI (Port 8000) ============
if [ "${FASTAPI:-on}" = "on" ]; then
  echo "🚀 Starte zentrale FastAPI (Port 8000)..."
  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/fastapi.log 2>&1 &
else
  echo "⏭️  FASTAPI=off – überspringe FastAPI."
fi

# ============ 🔷 Download/Init (OVI) ============
# 🚀 INIT-LOGIK (Dein Wunsch: Separater Skript-Start)
if [ "${INIT_SCRIPT:-off}" = "on" ]; then
  echo "🚀 Starte init.sh (Hintergrund)..."
  chmod +x /workspace/init.sh
  nohup bash /workspace/init.sh > /workspace/init_download.log 2>&1 & disown
fi

# ============ ✅ ABSCHLUSS ============
echo "✅ Dienste wurden gestartet (je nach config). Logs: /workspace/fastapi.log /workspace/jupyter.log"
tail -f /dev/null

