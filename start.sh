#!/bin/bash
set -euo pipefail




# ============ 🔧 Anti-Fragmentation für PyTorch ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
# (optional; falls root nicht erlaubt, diesen Block weglassen)
if [ -w /etc/profile.d ]; then
  echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' > /etc/profile.d/pytorch_alloc.sh || true
fi



# -------- tools.config nur laden, wenn vorhanden --------
if [ -f /workspace/tools.config ]; then
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