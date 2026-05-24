#!/bin/bash
set -euo pipefail

# 0) Self-heal
sed -i 's/\r$//' "$0" 2>/dev/null || true

# 1) Sicherstellen, dass tools.config im Volume existiert
mkdir -p /workspace /workspace/LTX-2/outputs
mkdir -p \
  /workspace/agent_runs \
  /workspace/exports \
  /workspace/jobs \
  /workspace/logs \
  /workspace/status \
  /workspace/status/model_downloads \
  /workspace/venvs

MODEL_DOWNLOAD_LOG="/workspace/logs/model_download.log"
MODEL_DOWNLOAD_MANIFEST_DIR="/workspace/status/model_downloads"
: > "$MODEL_DOWNLOAD_LOG"

function log_model() {
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "$MODEL_DOWNLOAD_LOG"
}

function fatal_init() {
  log_model "[fatal] $*"
  exit 1
}

INIT_LOCK_FILE="/workspace/status/init.lock"
exec 9>"$INIT_LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "[init] another init.sh is already running; exiting."
    exit 0
  fi
fi

if [ -f /app/tools.config ] && [ ! -f /workspace/tools.config ]; then
  cp -f /app/tools.config /workspace/tools.config
fi

sed -i 's/\r$//' /workspace/tools.config 2>/dev/null || true
python3 /workspace/scripts/apply_boot_model_profile.py
if [ -f /workspace/runtime/effective_tools.config ]; then
  sed -i 's/\r$//' /workspace/runtime/effective_tools.config 2>/dev/null || true
  source /workspace/runtime/effective_tools.config 2>/dev/null || true
else
  source /workspace/tools.config 2>/dev/null || true
fi
export PATH="/opt/venv/bin:/usr/local/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export GIT_TERMINAL_PROMPT=0
export GCM_INTERACTIVE=never
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HOME="/workspace/.cache/hf"
export HUGGINGFACE_HUB_CACHE="/workspace/.cache/hf/hub"
export HF_HUB_CACHE="/workspace/.cache/hf/hub"
export HF_ASSETS_CACHE="/workspace/.cache/hf/assets"
export TRANSFORMERS_CACHE="/workspace/.cache/hf/transformers"
export HF_XET_HIGH_PERFORMANCE="1"
unset HF_HUB_DISABLE_XET
export PYTHONUNBUFFERED=1
export PIP_NO_INPUT=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$TRANSFORMERS_CACHE"

function require_boot_profile() {
  python3 - <<'PY'
import json
from pathlib import Path

status_path = Path("/workspace/runtime/boot_model_profile_status.json")
profile_path = Path("/workspace/runtime/boot_model_profile.json")
if not status_path.is_file():
    raise SystemExit("BOOT_MODEL_PROFILE required: boot profile status missing")
status = json.loads(status_path.read_text())
if not status.get("loaded"):
    msg = status.get("error") or status.get("message") or "boot profile not loaded"
    raise SystemExit(f"BOOT_MODEL_PROFILE required: {msg}")
if not profile_path.is_file():
    raise SystemExit("BOOT_MODEL_PROFILE required: profile file missing")
profile = json.loads(profile_path.read_text())
required = profile.get("required_models")
if not isinstance(required, list) or not required:
    raise SystemExit("required_models empty")
print(" ".join(required))
PY
}

ACTIVE_REQUIRED_MODELS="$(require_boot_profile)" || fatal_init "BOOT_MODEL_PROFILE required"
log_model "[boot] profile active=/workspace/runtime/boot_model_profile.json"
log_model "[boot] required_models=${ACTIVE_REQUIRED_MODELS}"
log_model "[boot] tools: HiDream_O1_Dev=${HiDream_O1_Dev:-off} DW_LTX2=${DW_LTX2:-off} Ace_Step1_5=${Ace_Step1_5:-off} Qwen_TTS_Tokenizer=${Qwen_TTS_Tokenizer:-off} Qwen_TTS_Model=${Qwen_TTS_Model:-off} Qwen3_VL_Review=${Qwen3_VL_Review:-off} Vision_Review_Model=${Vision_Review_Model:-off}"

function model_required() {
  local NEEDLE="$1"
  case " $ACTIVE_REQUIRED_MODELS " in
    *" $NEEDLE "*) return 0 ;;
    *) return 1 ;;
  esac
}

function hf_xet_preflight() {
  python3 - <<'PY'
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

status = {
    "xet_required": True,
    "xet_preflight_ok": False,
    "high_performance_env": os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1",
    "errors": [],
}

def require(condition, message):
    if not condition:
        status["errors"].append(message)

try:
    hub = importlib.import_module("huggingface_hub")
    status["huggingface_hub_version"] = getattr(hub, "__version__", "unknown")
except Exception as exc:
    status["huggingface_hub_error"] = repr(exc)
    require(False, "huggingface_hub import failed")

try:
    xet = importlib.import_module("hf_xet")
    status["hf_xet_version"] = getattr(xet, "__version__", "unknown")
except Exception as exc:
    status["hf_xet_error"] = repr(exc)
    require(False, "hf_xet import failed")

require(os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1", "HF_XET_HIGH_PERFORMANCE must be 1")
require(os.environ.get("HF_HUB_DISABLE_XET") in (None, "", "0", "false", "False"), "HF_HUB_DISABLE_XET must not disable Xet")

for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"):
    value = os.environ.get(key)
    status[key] = value
    require(bool(value), f"{key} is not set")
    require(str(value).startswith("/workspace/"), f"{key} must point to /workspace")
    if value:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        require(os.access(path, os.W_OK), f"{key} is not writable: {value}")

free_gb = shutil.disk_usage("/workspace").free / (1024 ** 3)
status["workspace_free_gb"] = round(free_gb, 3)
min_free_gb = float(os.environ.get("DW_MIN_FREE_GB", "20"))
status["min_free_gb"] = min_free_gb
require(free_gb >= min_free_gb, f"workspace free space {free_gb:.1f} GB is below required {min_free_gb:.1f} GB")

status["xet_preflight_ok"] = not status["errors"]
Path("/workspace/status/hf_xet_preflight.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
if status["errors"]:
    print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit("HF/Xet preflight failed: " + "; ".join(status["errors"]))
print(json.dumps(status, sort_keys=True))
PY
}

HF_XET_PREFLIGHT_JSON="$(hf_xet_preflight)" || fatal_init "HF/Xet preflight failed"
log_model "[hf] preflight=${HF_XET_PREFLIGHT_JSON}"


# SoX systemweit sicherstellen
if ! command -v sox >/dev/null 2>&1; then
  if [ "${ALLOW_RUNTIME_APT:-0}" = "1" ]; then
    log_model "[sox] missing in image; ALLOW_RUNTIME_APT=1 enables legacy runtime install"
    apt-get update
    apt-get install -y sox libsox-fmt-all
  else
    fatal_init "sox missing; rebuild image with sox libsox-fmt-all or set ALLOW_RUNTIME_APT=1 for legacy runtime install"
  fi
fi


# 2) Download-Funktionen
function write_download_manifest() {
  local MODEL_ID="$1"
  local REPO_ID="$2"
  local TARGET="$3"
  local START_EPOCH="$4"
  local CACHE_HIT="$5"
  local STATUS="$6"
  local ERROR_MSG="${7:-}"
  MODEL_ID="$MODEL_ID" REPO_ID="$REPO_ID" TARGET="$TARGET" START_EPOCH="$START_EPOCH" CACHE_HIT="$CACHE_HIT" STATUS_VALUE="$STATUS" ERROR_MSG="$ERROR_MSG" python3 - <<'PY'
import json
import os
import time
from pathlib import Path

model_id = os.environ["MODEL_ID"]
repo_id = os.environ["REPO_ID"]
target = Path(os.environ["TARGET"])
start = float(os.environ["START_EPOCH"])
end = time.time()
files = []
total = 0
if target.exists():
    paths = [target] if target.is_file() else list(target.rglob("*"))
    for path in paths:
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append({"path": str(path), "bytes": size})
duration = max(end - start, 0.000001)
status = os.environ["STATUS_VALUE"]
payload = {
    "model_id": model_id,
    "repo_id": repo_id,
    "target": str(target),
    "files_expected": [item["path"] for item in files],
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
    "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)),
    "duration_sec": round(duration, 3),
    "total_bytes": total,
    "mb_per_sec": round((total / 1_000_000) / duration, 3),
    "mib_per_sec": round((total / (1024 * 1024)) / duration, 3),
    "cache_hit": os.environ["CACHE_HIT"] == "true",
    "xet_required": True,
    "xet_preflight_ok": True,
    "high_performance_env": os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1",
    "status": status,
}
if os.environ.get("ERROR_MSG"):
    payload["error"] = os.environ["ERROR_MSG"]
out = Path("/workspace/status/model_downloads") / f"{model_id}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

function verify_target_generic() {
  local TARGET="$1"
  [ -d "$TARGET" ] || return 1
  ! find "$TARGET" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . || return 1
  python3 - "$TARGET" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = [p for p in root.rglob("*") if p.is_file()]
if not files:
    raise SystemExit(1)
if any(p.stat().st_size <= 0 for p in files):
    raise SystemExit(1)
large = [p for p in files if p.suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"} and p.stat().st_size >= 1_000_000]
if not large and not (root / "config.json").is_file():
    raise SystemExit(1)
PY
}

function verify_expected_paths() {
  local MODEL_ID="$1"
  local TARGET="$2"
  shift 2
  [ -d "$TARGET" ] || return 1
  ! find "$TARGET" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . || return 1
  MODEL_ID="$MODEL_ID" TARGET="$TARGET" python3 - "$@" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["TARGET"])
errors = []
for spec in sys.argv[1:]:
    rel, _, min_bytes_raw = spec.partition(":")
    min_bytes = int(min_bytes_raw or "1")
    path = root / rel
    if not path.is_file():
        errors.append(f"missing {rel}")
        continue
    size = path.stat().st_size
    if size < min_bytes:
        errors.append(f"{rel} too small: {size} < {min_bytes}")
for path in root.rglob("*"):
    if path.is_file() and path.stat().st_size <= 0:
        errors.append(f"zero-byte file: {path}")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

function mark_model_ready() {
  local MODEL_ID="$1"
  local FLAG="$2"
  local TARGET="$3"
  shift 3
  local MANIFEST="$MODEL_DOWNLOAD_MANIFEST_DIR/$MODEL_ID.json"
  [ -f "$MANIFEST" ] || fatal_init "ready blocked for $MODEL_ID: manifest missing"
  python3 - "$MANIFEST" <<'PY' || fatal_init "ready blocked: manifest is not ready"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("status") not in {"ready", "skipped_cache_hit"}:
    raise SystemExit(1)
if not manifest.get("xet_preflight_ok") or not manifest.get("high_performance_env"):
    raise SystemExit(1)
PY
  verify_expected_paths "$MODEL_ID" "$TARGET" "$@" || fatal_init "ready blocked for $MODEL_ID: expected files failed size/existence checks"
  touch "$FLAG"
  log_model "[ready] model_id=$MODEL_ID flag=$FLAG"
}

function mark_manifest_ready_only() {
  local MODEL_ID="$1"
  local FLAG="$2"
  local MANIFEST="$MODEL_DOWNLOAD_MANIFEST_DIR/$MODEL_ID.json"
  [ -f "$MANIFEST" ] || fatal_init "ready blocked for $MODEL_ID: manifest missing"
  python3 - "$MANIFEST" <<'PY' || fatal_init "ready blocked for manifest-only model"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("status") not in {"ready", "skipped_cache_hit"}:
    raise SystemExit(1)
if not manifest.get("xet_preflight_ok") or not manifest.get("high_performance_env"):
    raise SystemExit(1)
if int(manifest.get("total_bytes") or 0) <= 1_000_000:
    raise SystemExit(1)
PY
  touch "$FLAG"
  log_model "[ready] model_id=$MODEL_ID flag=$FLAG"
}

function hf_download_all() {
  local MODEL_ID="$1"
  local REPO="$2"
  local START_EPOCH
  START_EPOCH="$(date +%s)"
  local TARGET="/workspace/.cache/hf/hub"
  local SNAPSHOT_PATH_FILE
  SNAPSHOT_PATH_FILE="$(mktemp)"
  if python3 - "$REPO" "$SNAPSHOT_PATH_FILE" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], Path(sys.argv[2])
path = snapshot_download(
    repo_id=repo,
    local_files_only=True,
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)
out.write_text(path)
PY
  then
    TARGET="$(cat "$SNAPSHOT_PATH_FILE")"
    rm -f "$SNAPSHOT_PATH_FILE"
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "true" "skipped_cache_hit"
    log_model "[hf] cache-hit model_id=$MODEL_ID repo=$REPO target=$TARGET"
    return 0
  fi
  rm -f "$SNAPSHOT_PATH_FILE"
  log_model "[hf] start model_id=$MODEL_ID repo=$REPO target=$TARGET xet=required high_performance=${HF_XET_HIGH_PERFORMANCE:-0}"
  SNAPSHOT_PATH_FILE="$(mktemp)"
  if python3 - "$SNAPSHOT_PATH_FILE" <<PY
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$REPO",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors","*.json","*.txt"]
)
Path(sys.argv[1]).write_text(path)
PY
  then
    TARGET="$(cat "$SNAPSHOT_PATH_FILE")"
    rm -f "$SNAPSHOT_PATH_FILE"
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "false" "ready"
    log_model "[hf] done model_id=$MODEL_ID repo=$REPO"
    return 0
  fi

  rm -f "$SNAPSHOT_PATH_FILE"
  write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "false" "failed" "download failed"
  log_model "[hf] fail model_id=$MODEL_ID repo=$REPO"
  return 1
}

function hf_download_file() {
  local MODEL_ID="$1"
  local REPO="$2"
  local FILE="$3"
  local TARGET="$4"
  local START_EPOCH
  START_EPOCH="$(date +%s)"
  if [ -s "$TARGET/$FILE" ]; then
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET/$FILE" "$START_EPOCH" "true" "skipped_cache_hit"
    log_model "[hf] cache-hit model_id=$MODEL_ID file=$TARGET/$FILE"
    return 0
  fi

  log_model "[hf] start model_id=$MODEL_ID repo=$REPO file=$FILE target=$TARGET xet=required high_performance=${HF_XET_HIGH_PERFORMANCE:-0}"
  if python3 - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$TARGET", local_dir_use_symlinks=False, resume_download=True)
PY
  then
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET/$FILE" "$START_EPOCH" "false" "ready"
    log_model "[hf] done model_id=$MODEL_ID file=$TARGET/$FILE"
    return 0
  fi

  write_download_manifest "$MODEL_ID" "$REPO" "$TARGET/$FILE" "$START_EPOCH" "false" "failed" "download failed"
  log_model "[hf] fail model_id=$MODEL_ID file=$REPO/$FILE"
  return 1
}

function hf_download_snapshot_to_dir() {
  local MODEL_ID="$1"
  local REPO="$2"
  local TARGET="$3"
  local START_EPOCH
  START_EPOCH="$(date +%s)"
  if verify_target_generic "$TARGET"; then
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "true" "skipped_cache_hit"
    log_model "[hf] cache-hit model_id=$MODEL_ID target=$TARGET"
    return 0
  fi

  log_model "[hf] start model_id=$MODEL_ID repo=$REPO target=$TARGET xet=required high_performance=${HF_XET_HIGH_PERFORMANCE:-0}"
  if python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    local_dir="$TARGET",
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
  then
    verify_target_generic "$TARGET" || {
      write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "false" "failed" "generic target verification failed"
      log_model "[hf] fail model_id=$MODEL_ID target=$TARGET verify=generic"
      return 1
    }
    write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "false" "ready"
    log_model "[hf] done model_id=$MODEL_ID target=$TARGET"
    return 0
  fi

  write_download_manifest "$MODEL_ID" "$REPO" "$TARGET" "$START_EPOCH" "false" "failed" "download failed"
  log_model "[hf] fail model_id=$MODEL_ID repo=$REPO target=$TARGET"
  return 1
}

function gemma_model_ready() {
  local TARGET="$1"
  [ -f "$TARGET/config.json" ] || return 1
  [ -f "$TARGET/tokenizer.model" ] || return 1
  [ -f "$TARGET/tokenizer.json" ] || return 1
  [ -f "$TARGET/tokenizer_config.json" ] || return 1
  [ -f "$TARGET/preprocessor_config.json" ] || return 1
  [ -f "$TARGET/model.safetensors.index.json" ] || return 1
  ! find "$TARGET" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . || return 1
  python3 - "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
for shard in set(index.get("weight_map", {}).values()):
    path = root / shard
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(1)
for path in root.rglob("*"):
    if path.is_file() and path.stat().st_size == 0:
        raise SystemExit(1)
PY
}

MODELS_DIR="/workspace/LTX-2/checkpoints"
LORA_DIR="$MODELS_DIR/loras"
QWEN_MODELS_DIR="/workspace/models/qwen3-tts"
DEFAULT_AUDIO_VENV="/opt/venv"
LEGACY_QWEN_VENV="/workspace/venvs/qwen3-tts"
QWEN_PREBUILT_VENV="/opt/venvs/qwen3-tts"
QWEN_RUNTIME_FLAG="/workspace/status/qwen_tts_runtime_ready"
ACE_STEP_ROOT="/workspace/ACE-Step-1.5"
ACE_STEP_CKPT_DIR="$ACE_STEP_ROOT/checkpoints"
ACE_STEP_READY_FLAG="/workspace/status/ace_step_ready"
ACE_STEP_ENV_FLAG="/workspace/status/ace_step_env_ready"
Qwen3_VL_Review="${Qwen3_VL_Review:-off}"
Vision_Review_Model="${Vision_Review_Model:-off}"
QWEN3_VL_MODEL_DIR="/workspace/models/Qwen3-VL-4B-Instruct-FP8"
QWEN3_VL_READY_FLAG="/workspace/status/qwen3_vl_ready"
mkdir -p "$MODELS_DIR/ltx-2.3" "$MODELS_DIR/gemma-3" "$LORA_DIR"
mkdir -p "$QWEN_MODELS_DIR"
mkdir -p "$ACE_STEP_CKPT_DIR"
mkdir -p /workspace/scripts

USE_LEGACY_QWEN_VENV="${USE_LEGACY_QWEN_VENV:-0}"
if [ "$USE_LEGACY_QWEN_VENV" = "1" ]; then
  QWEN_VENV="${QWEN_VENV:-$LEGACY_QWEN_VENV}"
  QWEN_PYTHON="${QWEN_PYTHON:-$QWEN_VENV/bin/python}"
  ACE_STEP_PYTHON="${ACE_STEP_PYTHON:-$QWEN_PYTHON}"
else
  QWEN_VENV="${QWEN_VENV:-$DEFAULT_AUDIO_VENV}"
  QWEN_PYTHON="${QWEN_PYTHON:-$DEFAULT_AUDIO_VENV/bin/python}"
  ACE_STEP_PYTHON="${ACE_STEP_PYTHON:-$DEFAULT_AUDIO_VENV/bin/python}"
fi
export QWEN_VENV QWEN_PYTHON ACE_STEP_PYTHON USE_LEGACY_QWEN_VENV

# ----------------------------------------------------
# 4. Qwen3-TTS / Shared Runtime Sektion
# ----------------------------------------------------
if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ] || [ "${Ace_Step1_5:-off}" = "on" ]; then
  echo "[qwen] Preparing runtime environment..."
  if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
    rm -f "$QWEN_RUNTIME_FLAG"
  fi
  if [ "${Ace_Step1_5:-off}" = "on" ]; then
    rm -f "$ACE_STEP_ENV_FLAG"
  fi

  if [ "$USE_LEGACY_QWEN_VENV" = "1" ]; then
    echo "[qwen] USE_LEGACY_QWEN_VENV=1: enabling legacy /workspace qwen venv build."
    mkdir -p "/workspace/venvs"
    if [ ! -e "$QWEN_VENV" ] && [ -d "$QWEN_PREBUILT_VENV" ]; then
      ln -sfn "$QWEN_PREBUILT_VENV" "$QWEN_VENV"
    elif [ ! -f "$QWEN_VENV/bin/activate" ]; then
      python3 -m venv "$QWEN_VENV"
    fi

    if [ ! -d "$QWEN_PREBUILT_VENV" ] && \
       { [ "${Qwen_TTS_Tokenizer:-off}" != "on" ] && [ "${Qwen_TTS_Model:-off}" != "on" ] || [ ! -f "/workspace/status/qwen_tts_env_ready" ]; }; then
      "$QWEN_VENV/bin/pip" install --no-cache-dir -U pip setuptools wheel
      "$QWEN_VENV/bin/pip" install --no-cache-dir ninja packaging psutil pybind11
      "$QWEN_VENV/bin/pip" install --no-cache-dir \
        torch==2.7.0 \
        torchaudio==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128

      "$QWEN_VENV/bin/pip" install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation
      if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
        "$QWEN_VENV/bin/pip" install --no-cache-dir \
          "qwen-tts==0.1.1" \
          "transformers==4.57.3" \
          "accelerate==1.12.0"
      fi
    fi
  else
    echo "[qwen] Using consolidated audio runtime: $QWEN_PYTHON"
    [ "$QWEN_PYTHON" != "$LEGACY_QWEN_VENV/bin/python" ] || fatal_init "legacy qwen venv is not allowed unless USE_LEGACY_QWEN_VENV=1"
    [ "$ACE_STEP_PYTHON" != "$LEGACY_QWEN_VENV/bin/python" ] || fatal_init "legacy ACE python is not allowed unless USE_LEGACY_QWEN_VENV=1"
    [ -x "$QWEN_PYTHON" ] || fatal_init "QWEN_PYTHON is not executable: $QWEN_PYTHON"
    [ -x "$ACE_STEP_PYTHON" ] || fatal_init "ACE_STEP_PYTHON is not executable: $ACE_STEP_PYTHON"
  fi

  if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
    "$QWEN_PYTHON" - <<'PY'
import qwen_tts
import transformers
print("qwen_tts ok", qwen_tts.__file__)
print("transformers", transformers.__version__)
PY
    touch "/workspace/status/qwen_tts_env_ready"
    touch "$QWEN_RUNTIME_FLAG"
  fi
fi

if [ "${Ace_Step1_5:-off}" = "on" ]; then
  echo "[ace-step] Preparing shared audio runtime..."

  if [ "$USE_LEGACY_QWEN_VENV" = "1" ] && [ ! -f "$ACE_STEP_ENV_FLAG" ] && ! "$ACE_STEP_PYTHON" - <<'PY'
import transformers.configuration_utils as cu
import diffusers
import loguru
import toml
import modelscope
import torchvision
import torchao
assert hasattr(cu, "layer_type_validation")
PY
  then
    "$QWEN_VENV/bin/pip" install --no-cache-dir \
      "torchvision==0.22.0" \
      "diffusers" \
      "toml" \
      "loguru" \
      "modelscope" \
      "torchao" \
      "matplotlib>=3.7.5" \
      "scipy>=1.10.1" \
      "soundfile>=0.13.1" \
      "einops>=0.8.1" \
      "fastapi>=0.110.0" \
      "uvicorn[standard]>=0.27.0" \
      "numba>=0.63.1" \
      "vector-quantize-pytorch>=1.27.15" \
      "python-dotenv" \
      "xxhash"

    if [ -d "$ACE_STEP_ROOT/acestep/third_parts/nano-vllm" ]; then
      "$QWEN_VENV/bin/pip" install --no-cache-dir -e "$ACE_STEP_ROOT/acestep/third_parts/nano-vllm"
    fi
  fi

  PYTHONPATH="$ACE_STEP_ROOT:${PYTHONPATH:-}" "$ACE_STEP_PYTHON" - <<'PY'
import transformers
import transformers.configuration_utils as cu
import diffusers
import loguru
import toml
import modelscope
import torchvision
import torch
import torchao
import acestep
import acestep.handler
import acestep.inference
import acestep.llm_inference

assert hasattr(cu, "layer_type_validation")
print("ace-step runtime ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("torchvision", torchvision.__version__)
PY
  touch "$ACE_STEP_ENV_FLAG"
fi

if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] || [ "${Qwen_TTS_Model:-off}" = "on" ]; then
  touch "/workspace/status/qwen_tts_env_ready"
  touch "$QWEN_RUNTIME_FLAG"
fi

if [ "${Qwen_TTS_Tokenizer:-off}" = "on" ] && model_required "qwen_tts"; then
  echo "[qwen] Qwen_TTS_Tokenizer is ON. Checking tokenizer path..."
  if [ -d "/opt/models/qwen/Qwen3-TTS-Tokenizer-12Hz" ]; then
    ln -sfn "/opt/models/qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz"
    write_download_manifest "qwen_tts_tokenizer" "local:/opt/models/qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz" "$(date +%s)" "true" "skipped_cache_hit"
  elif [ ! -f "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz/config.json" ]; then
    hf_download_snapshot_to_dir "qwen_tts_tokenizer" "Qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz"
  else
    hf_download_snapshot_to_dir "qwen_tts_tokenizer" "Qwen/Qwen3-TTS-Tokenizer-12Hz" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz"
  fi

  mark_model_ready "qwen_tts_tokenizer" "/workspace/status/qwen_tts_tokenizer_ready" "$QWEN_MODELS_DIR/Qwen3-TTS-Tokenizer-12Hz" \
    "config.json:1000" \
    "model.safetensors:600000000"
fi

if [ "${Qwen_TTS_Model:-off}" = "on" ] && model_required "qwen_tts"; then
  echo "[qwen] Qwen_TTS_Model is ON. Checking model path..."
  if [ -d "/opt/models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" ]; then
    ln -sfn "/opt/models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    write_download_manifest "qwen_tts_model" "local:/opt/models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$(date +%s)" "true" "skipped_cache_hit"
  elif [ ! -f "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json" ]; then
    hf_download_snapshot_to_dir "qwen_tts_model" "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  else
    hf_download_snapshot_to_dir "qwen_tts_model" "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  fi

  mark_model_ready "qwen_tts_model" "/workspace/status/qwen_tts_model_ready" "$QWEN_MODELS_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice" \
    "config.json:1000" \
    "model.safetensors:3000000000" \
    "speech_tokenizer/config.json:1000" \
    "speech_tokenizer/model.safetensors:600000000"
fi

# ----------------------------------------------------
# 5. ACE-Step 1.5 Sektion
# ----------------------------------------------------
if [ "${Ace_Step1_5:-off}" = "on" ] && model_required "ace_step"; then
  echo "[ace-step] Ace_Step1_5 is ON. Checking model path..."
  if [ -f "$ACE_STEP_CKPT_DIR/acestep-v15-turbo/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/vae/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/Qwen3-Embedding-0.6B/config.json" ] && \
     [ -f "$ACE_STEP_CKPT_DIR/acestep-5Hz-lm-1.7B/config.json" ]; then
    echo "[ace-step] Main model already present."
    write_download_manifest "ace_step" "ACE-Step/Ace-Step1.5" "$ACE_STEP_CKPT_DIR" "$(date +%s)" "true" "skipped_cache_hit"
  else
    hf_download_snapshot_to_dir "ace_step" "ACE-Step/Ace-Step1.5" "$ACE_STEP_CKPT_DIR"
  fi

  mark_model_ready "ace_step" "$ACE_STEP_READY_FLAG" "$ACE_STEP_CKPT_DIR" \
    "acestep-v15-turbo/config.json:100" \
    "vae/config.json:100" \
    "Qwen3-Embedding-0.6B/config.json:100" \
    "acestep-5Hz-lm-1.7B/config.json:100"
fi

# ----------------------------------------------------
# 6. HiDream-O1-Dev Sektion
# ----------------------------------------------------
if [ "${HiDream_O1_Dev:-off}" = "on" ] && model_required "hidream"; then
  echo "[hidream] HiDream_O1_Dev is ON. Checking cache..."
  if hf_download_all "hidream" "${HIDREAM_O1_DEV_REPO:-HiDream-ai/HiDream-O1-Image-Dev}"; then
    mark_manifest_ready_only "hidream" "/workspace/status/hidream_ready"
  fi
fi

# ----------------------------------------------------
# 7. LTX-2 Basis-Modelle
# ----------------------------------------------------
if [ "${DW_LTX2:-off}" = "on" ] && model_required "ltx2"; then
  if [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-dev.safetensors" ] || \
     [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" ] || \
     [ ! -f "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors" ] || \
     ! gemma_model_ready "$MODELS_DIR/gemma-3"; then
    echo "🚀 Hauptmodelle fehlen – Starte Setup..."

    if [ -n "${HF_TOKEN:-}" ]; then
      python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    fi

    hf_download_file "ltx2_base" "Lightricks/LTX-2.3" "ltx-2.3-22b-dev.safetensors" "$MODELS_DIR/ltx-2.3"
    hf_download_file "ltx2_upscaler" "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2.3"
    hf_download_file "ltx2_lora" "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2.3"

    echo "🚀 Lade Gemma-3..."
    hf_download_snapshot_to_dir "ltx2_gemma3" "google/gemma-3-12b-it-qat-q4_0-unquantized" "$MODELS_DIR/gemma-3"
  else
    write_download_manifest "ltx2_base" "Lightricks/LTX-2.3" "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-dev.safetensors" "$(date +%s)" "true" "skipped_cache_hit"
    write_download_manifest "ltx2_upscaler" "Lightricks/LTX-2.3" "$MODELS_DIR/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$(date +%s)" "true" "skipped_cache_hit"
    write_download_manifest "ltx2_lora" "Lightricks/LTX-2.3" "$MODELS_DIR/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors" "$(date +%s)" "true" "skipped_cache_hit"
    write_download_manifest "ltx2_gemma3" "google/gemma-3-12b-it-qat-q4_0-unquantized" "$MODELS_DIR/gemma-3" "$(date +%s)" "true" "skipped_cache_hit"
  fi
  verify_expected_paths "ltx2_base" "$MODELS_DIR/ltx-2.3" \
    "ltx-2.3-22b-dev.safetensors:1000000000" \
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors:1000000" \
    "ltx-2.3-22b-distilled-lora-384.safetensors:1000000" || fatal_init "ltx2 expected file checks failed"
  gemma_model_ready "$MODELS_DIR/gemma-3" || fatal_init "ltx2 gemma ready check failed"
else
  echo "⏭️  DW_LTX2=off – überspringe LTX-2 Basis-Modelle."
fi

# ----------------------------------------------------
# 8. Optional Qwen3-VL Vision Review Modell
# ----------------------------------------------------
if { [ "$Qwen3_VL_Review" = "on" ] || [ "$Vision_Review_Model" = "on" ]; } && model_required "qwen3_vl_review"; then
  echo "[qwen3-vl] Qwen3-VL review model is ON. Checking model path..."
  rm -f "$QWEN3_VL_READY_FLAG"
  if python3 /workspace/scripts/download_qwen3_vl_model.py; then
    touch "$QWEN3_VL_READY_FLAG"
    echo "[qwen3-vl] Model ready at $QWEN3_VL_MODEL_DIR"
  else
    rm -f "$QWEN3_VL_READY_FLAG"
    echo "[qwen3-vl] ERROR: model download/verify failed."
  fi
else
  rm -f "$QWEN3_VL_READY_FLAG"
fi

# ----------------------------------------------------
# 9. LoRA Sektion
# ----------------------------------------------------
echo "📥 Prüfe LoRA Downloads..."

[[ "${Lora1:-off}" == "on" && ! -f "$LORA_DIR/ltx2-cakeify-v2.safetensors" ]] && \
  hf_download_file "lora1" "kabachuha/ltx2-cakeify" "ltx2-cakeify-v2.safetensors" "$LORA_DIR"

[[ "${Lora2:-off}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-ic-lora-detailer.safetensors" ]] && \
  hf_download_file "lora2" "Lightricks/LTX-2-19b-IC-LoRA-Detailer" "ltx-2-19b-ic-lora-detailer.safetensors" "$LORA_DIR"

[[ "${Lora3:-off}" == "on" && ! -f "$LORA_DIR/ltx-2-19b-lora-camera-control-static.safetensors" ]] && \
  hf_download_file "lora3" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" "ltx-2-19b-lora-camera-control-static.safetensors" "$LORA_DIR"

[[ "${Lora4:-off}" == "on" && ! -f "$LORA_DIR/LTX-2-Image2Vid-Adapter.safetensors" ]] && \
  hf_download_file "lora4" "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa" "LTX-2-Image2Vid-Adapter.safetensors" "$LORA_DIR"

# ----------------------------------------------------
# 10. Abschluss
# ----------------------------------------------------
chmod -R 777 "$MODELS_DIR" || true
chmod -R 777 "$QWEN_MODELS_DIR" || true
if [ "$USE_LEGACY_QWEN_VENV" = "1" ]; then
  chmod -R 777 "$QWEN_VENV" || true
fi
chmod -R 777 "$ACE_STEP_CKPT_DIR" || true
chmod +x /workspace/scripts/*.sh 2>/dev/null || true
chmod +x /workspace/scripts/*.py 2>/dev/null || true
echo "🏁 init.sh erfolgreich beendet."
touch /workspace/status/init_done

# ----------------------------------------------------
# 11. Optional: Real-ESRGAN AI Installer (nach init_done)
# ----------------------------------------------------
if [ "${UPSCALER_INSTALL:-off}" = "on" ]; then
  UPSCALER_INSTALLER="/workspace/upscaler_installer_minimal/install_realesrgan_ai_pod.sh"
  if [ -f "$UPSCALER_INSTALLER" ]; then
    chmod +x "$UPSCALER_INSTALLER" 2>/dev/null || true
    echo "🛠️ Starte optionalen Upscaler-Installer..."
    if bash "$UPSCALER_INSTALLER"; then
      touch /workspace/status/upscaler_install_done
      echo "✅ Upscaler-Installer erfolgreich."
    else
      touch /workspace/status/upscaler_install_failed
      echo "❌ Upscaler-Installer fehlgeschlagen."
    fi
  else
    echo "⚠️ Upscaler-Installer nicht gefunden: $UPSCALER_INSTALLER"
 



 fi
fi
