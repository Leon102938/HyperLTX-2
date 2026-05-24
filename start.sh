#!/usr/bin/env bash
set -u

# ============================================================
# Content Maschine start.sh
# - Jupyter sofort starten
# - FastAPI sofort starten
# - init.sh / Auto-DW im Hintergrund starten
# - Container am Leben halten
# ============================================================

mkdir -p /workspace/logs /workspace/status /workspace/runtime

LOG_START="/workspace/logs/start.log"
LOG_JUPYTER="/workspace/logs/jupyter.log"
LOG_API="/workspace/logs/api.log"
LOG_INIT="/workspace/logs/init.log"

# Legacy log links, falls alte RunPod/UI-Pfade darauf schauen
ln -sf "$LOG_JUPYTER" /workspace/jupyter.log 2>/dev/null || true
ln -sf "$LOG_API" /workspace/fastapi.log 2>/dev/null || true
ln -sf "$LOG_INIT" /workspace/init_download.log 2>/dev/null || true

log() {
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') [start] $*" | tee -a "$LOG_START"
}

log "start.sh begin"

# ============ CUDA / HF ENV ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export HF_HUB_DISABLE_TELEMETRY=1

# hf_xet ist unser finaler Speed-Pfad
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_DISABLE_XET || true

# hf_transfer kann installiert bleiben, ist aber nicht der primäre Pfad
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

export HF_HOME="${HF_HOME:-/workspace/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/hf/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/hf/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/workspace/.cache/hf/assets}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE"

log "BOOT_MODEL_PROFILE=${BOOT_MODEL_PROFILE:-}"
log "CONTENT_MACHINE_BOOT_PROFILE_B64=${CONTENT_MACHINE_BOOT_PROFILE_B64:+set}"
log "USE_OPT_VENV_FOR_AUDIO=${USE_OPT_VENV_FOR_AUDIO:-1}"
log "HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-}"
log "HF_HOME=${HF_HOME}"
log "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"

# Audio final default
export USE_OPT_VENV_FOR_AUDIO="${USE_OPT_VENV_FOR_AUDIO:-1}"

# Optional global shell hint
if [ -w /etc/profile.d ]; then
  echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' > /etc/profile.d/pytorch_alloc.sh 2>/dev/null || true
fi

# ============ Boot Profile anwenden ============
# Wichtig:
# - Jupyter/API dürfen NICHT blockieren, wenn kein Profil gesetzt ist.
# - init.sh erzwingt später selbst BOOT_MODEL_PROFILE.
if [ -f /workspace/scripts/apply_boot_model_profile.py ]; then
  if [ -n "${BOOT_MODEL_PROFILE:-}" ] || [ -n "${CONTENT_MACHINE_BOOT_PROFILE_B64:-}" ]; then
    log "applying boot model profile"
    /opt/venv/bin/python /workspace/scripts/apply_boot_model_profile.py >> "$LOG_START" 2>&1 || {
      log "WARNING: apply_boot_model_profile.py failed; init.sh may fail later"
    }
  else
    log "no BOOT_MODEL_PROFILE/CONTENT_MACHINE_BOOT_PROFILE_B64 set; skipping profile apply in start.sh"
  fi
fi

# Optional effective config laden, aber START_* bleibt unabhängig davon.
if [ -f /workspace/runtime/effective_tools.config ]; then
  # shellcheck disable=SC1091
  source /workspace/runtime/effective_tools.config || true
elif [ -f /workspace/tools.config ]; then
  # shellcheck disable=SC1091
  source /workspace/tools.config || true
fi

# ============ BASE_URL setzen ============
log "detecting RunPod proxy URL"
POD_ID_VALUE="${RUNPOD_POD_ID:-${POD_ID:-}}"
if [ -n "$POD_ID_VALUE" ]; then
  BASE_URL="https://${POD_ID_VALUE}-8000.proxy.runpod.net"
  export BASE_URL
  echo "BASE_URL=$BASE_URL" > /workspace/.env
  log "BASE_URL=$BASE_URL"
else
  log "RUNPOD_POD_ID not set; /workspace/.env BASE_URL not written"
fi

# ============ JupyterLab Dark Theme ============
mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
cat > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings <<'EOF'
{
  "theme": "JupyterLab Dark"
}
EOF

# ============ Service switches ============
# Neue robuste Defaults: Services starten standardmäßig.
# Zum Deaktivieren explizit START_JUPYTER=off / START_FASTAPI=off / START_INIT=off setzen.
START_JUPYTER="${START_JUPYTER:-on}"
START_FASTAPI="${START_FASTAPI:-on}"
START_INIT="${START_INIT:-on}"

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"

# ============ Helpers ============
port_probe() {
  /opt/venv/bin/python - <<PY
import socket
for port in (${FASTAPI_PORT}, ${JUPYTER_PORT}):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        print(f"{port} OPEN")
    except Exception as e:
        print(f"{port} CLOSED {e!r}")
    finally:
        s.close()
PY
}

start_jupyter() {
  if [ "$START_JUPYTER" != "on" ]; then
    log "START_JUPYTER=$START_JUPYTER; skipping Jupyter"
    return 0
  fi

  if pgrep -f "jupyter-lab.*${JUPYTER_PORT}|jupyter lab.*${JUPYTER_PORT}" >/dev/null 2>&1; then
    log "Jupyter already running on port ${JUPYTER_PORT}"
    return 0
  fi

  log "starting JupyterLab on port ${JUPYTER_PORT}"

  nohup /opt/venv/bin/jupyter lab \
    --ServerApp.ip=0.0.0.0 \
    --ServerApp.port="${JUPYTER_PORT}" \
    --ServerApp.open_browser=False \
    --ServerApp.allow_root=True \
    --ServerApp.root_dir=/workspace \
    --ServerApp.token='' \
    --IdentityProvider.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' \
    > "$LOG_JUPYTER" 2>&1 &

  echo $! > /workspace/runtime/jupyter.pid
  log "Jupyter PID=$(cat /workspace/runtime/jupyter.pid) log=$LOG_JUPYTER"
}

start_api() {
  if [ "$START_FASTAPI" != "on" ]; then
    log "START_FASTAPI=$START_FASTAPI; skipping FastAPI"
    return 0
  fi

  if pgrep -f "uvicorn app.main:app.*${FASTAPI_PORT}|uvicorn.*app.main:app" >/dev/null 2>&1; then
    log "FastAPI already running on port ${FASTAPI_PORT}"
    return 0
  fi

  log "starting FastAPI on port ${FASTAPI_PORT}"

  nohup /opt/venv/bin/uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${FASTAPI_PORT}" \
    > "$LOG_API" 2>&1 &

  echo $! > /workspace/runtime/api.pid
  log "FastAPI PID=$(cat /workspace/runtime/api.pid) log=$LOG_API"
}

start_init() {
  if [ "$START_INIT" != "on" ]; then
    log "START_INIT=$START_INIT; skipping init.sh / Auto-DW"
    return 0
  fi

  if pgrep -f "bash /workspace/init.sh|/workspace/init.sh" >/dev/null 2>&1; then
    log "init.sh already running"
    return 0
  fi

  if [ -z "${BOOT_MODEL_PROFILE:-}" ] && [ -z "${CONTENT_MACHINE_BOOT_PROFILE_B64:-}" ]; then
    log "no BOOT_MODEL_PROFILE/CONTENT_MACHINE_BOOT_PROFILE_B64 set; init.sh not started"
    log "set BOOT_MODEL_PROFILE=full for Auto-DW"
    return 0
  fi

  log "starting init.sh / Auto-DW in background"

  chmod +x /workspace/init.sh || true

  nohup bash /workspace/init.sh \
    > "$LOG_INIT" 2>&1 &

  echo $! > /workspace/runtime/init.pid
  log "init.sh PID=$(cat /workspace/runtime/init.pid) log=$LOG_INIT"
}

# ============ Start order ============
# UI/API zuerst, damit RunPod Launch / Hermes nicht hängen.
start_jupyter
start_api
start_init

sleep 3

log "process list:"
ps -eo pid,ppid,etime,cmd | egrep 'jupyter|uvicorn|init.sh|start.sh' | grep -v egrep | tee -a "$LOG_START" || true

log "port probe:"
port_probe | tee -a "$LOG_START" || true

log "services launched; container keepalive active"
log "logs: $LOG_START $LOG_INIT $LOG_API $LOG_JUPYTER"

# Container alive halten und Logs live anzeigen.
tail -F "$LOG_START" "$LOG_INIT" "$LOG_API" "$LOG_JUPYTER"