#!/bin/bash
set -euo pipefail

# ============ 📂 Pfad-Logik (Automatisches Finden) ============
# Git klont oft in /workspace/HyperLTX-2, RunPod erwartet oft /workspace
if [ -d "/workspace/HyperLTX-2" ]; then
    PROJECT_ROOT="/workspace/HyperLTX-2"
elif [ -d "/workspace/LTX-2" ]; then
    PROJECT_ROOT="/workspace/LTX-2"
else
    PROJECT_ROOT="/workspace"
fi

echo "📂 Project Root gesetzt auf: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ============ 🔧 Anti-Fragmentation für PyTorch ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# -------- tools.config nur laden, wenn vorhanden --------
if [ -f "./tools.config" ]; then
  source ./tools.config
fi

# 🌍 BASE_URL automatisch setzen
echo "🌐 Ermittle dynamische RunPod Proxy-URL..."
POD_ID="${RUNPOD_POD_ID:-}"
if [ -z "$POD_ID" ]; then
  echo "⚠️ WARNUNG: RUNPOD_POD_ID nicht gesetzt."
else
  BASE_URL="https://${POD_ID}-8000.proxy.runpod.net"
  export BASE_URL
  echo "BASE_URL=$BASE_URL" > "$PROJECT_ROOT/.env"
  echo "✅ BASE_URL erfolgreich gesetzt: $BASE_URL"
fi

# -------- 🧠 LTX-2 PYTHONPATH (Core Auswahl) --------
# Hier binden wir die ltx-core und ltx-pipelines Pakete ein
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/packages/ltx-core/src:$PROJECT_ROOT/packages/ltx-pipelines/src:${PYTHONPATH:-}"
echo "🐍 PYTHONPATH gesetzt auf: $PYTHONPATH"

# ============ 🔷 JUPYTERLAB THEME ============
mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
echo '{ "theme": "JupyterLab Dark" }' > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# ============ 🔷 JUPYTERLAB (Port 8888) ============
if [ "${JUPYTER:-off}" = "on" ]; then
  echo "🧠 Starte JupyterLab..."
  nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.disable_check_xsrf=True --NotebookApp.notebook_dir='/workspace' --ServerApp.allow_origin='*' > /workspace/jupyter.log 2>&1 &
fi

# ============ 🔷 FASTAPI (Port 8000) ============
if [ "${FASTAPI:-on}" = "on" ]; then
  echo "🚀 Starte zentrale FastAPI aus $PROJECT_ROOT..."
  # Wir starten uvicorn direkt aus dem Projektordner, damit app.main gefunden wird
  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/fastapi.log 2>&1 &
else
  echo "⏭️  FASTAPI=off"
fi

# ============ 🔷 Download/Init (OVI) ============
if [ "${INIT_SCRIPT:-off}" = "on" ]; then
  echo "🚀 Starte init.sh..."
  # Falls init.sh im Repo liegt, machen wir es ausführbar
  [ -f "./init.sh" ] && chmod +x ./init.sh && nohup bash ./init.sh > /workspace/init_download.log 2>&1 & disown
fi

# ============ ✅ ABSCHLUSS ============
echo "✅ Dienste gestartet. Logs: /workspace/fastapi.log /workspace/jupyter.log"
# Verhindert, dass der Container sich beendet
tail -f /dev/null