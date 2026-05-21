#!/bin/bash
set -euo pipefail




# ============ 🔧 Anti-Fragmentation für PyTorch ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HOME="${HF_HOME:-/workspace/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/hf/hub}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
echo "[hf] HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
echo "[hf] HF_HOME=${HF_HOME:-}"
echo "[hf] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-}"
# (optional; falls root nicht erlaubt, diesen Block weglassen)
if [ -w /etc/profile.d ]; then
  echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' > /etc/profile.d/pytorch_alloc.sh || true
fi



# -------- Boot Profile anwenden, dann effective/default tools.config laden --------
if [ -f /workspace/scripts/apply_boot_model_profile.py ]; then
  python3 /workspace/scripts/apply_boot_model_profile.py || true
fi

if [ -f /workspace/runtime/effective_tools.config ]; then
  source /workspace/runtime/effective_tools.config
elif [ -f /workspace/tools.config ]; then
  source /workspace/tools.config
fi





# 🌍 BASE_URL automatisch setzen (RUNPOD_POD_ID sicher expandieren)
echo "🌐 Ermittle dynamische RunPod Proxy-URL..."
POD_ID="${RUNPOD_POD_ID:-${POD_ID:-}}"
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
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.root_dir=/workspace \
    --ServerApp.allow_origin='*' \
    > /workspace/jupyter.log 2>&1 &
  echo "✅ Jupyter gestartet. Log: /workspace/jupyter.log"
else
  echo "⏭️  JUPYTER=off – überspringe Jupyter."
fi




# ============ 🔷 FASTAPI (Port 8000) ============
if [ "${FASTAPI:-off}" = "on" ]; then
  echo "🚀 Starte zentrale FastAPI (Port 8000)..."
  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/fastapi.log 2>&1 &
else
  echo "⏭️  FASTAPI=off – überspringe FastAPI."
fi

# ============ 🔷 Download/Init (LTX) ============
# 🚀 INIT-LOGIK (Dein Wunsch: Separater Skript-Start)
if [ "${INIT_SCRIPT:-off}" = "on" ]; then
  echo "🚀 Starte init.sh (Hintergrund)..."
  chmod +x /workspace/init.sh
  nohup bash /workspace/init.sh > /workspace/init_download.log 2>&1 &
fi


# ============ ✅ ABSCHLUSS ============
echo "✅ Dienste wurden gestartet (je nach config). Logs: /workspace/fastapi.log /workspace/jupyter.log"
tail -f /dev/null
