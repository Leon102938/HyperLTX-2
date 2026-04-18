#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"

for config_file in \
  "$ROOT_DIR/config/director_llm.env" \
  "$ROOT_DIR/config/director_llm.env.local"
do
  if [ -f "$config_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSURE_LLAMA_CPP="$SCRIPT_DIR/ensure_llama_cpp.sh"
DOWNLOAD_MODEL="$SCRIPT_DIR/download_director_model.py"

DIRECTOR_LLM_HOST="${DIRECTOR_LLM_HOST:-127.0.0.1}"
DIRECTOR_LLM_PORT="${DIRECTOR_LLM_PORT:-8011}"
DIRECTOR_LLM_BASE_URL="${DIRECTOR_LLM_BASE_URL:-http://${DIRECTOR_LLM_HOST}:${DIRECTOR_LLM_PORT}}"
DIRECTOR_LLM_MODEL_DIR="${DIRECTOR_LLM_MODEL_DIR:-$ROOT_DIR/models/director/qwen3.6-35b-a3b/gguf}"
DIRECTOR_LLM_MODEL_FILE="${DIRECTOR_LLM_MODEL_FILE:-Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf}"
DIRECTOR_LLM_MODEL_PATH="${DIRECTOR_LLM_MODEL_PATH:-$DIRECTOR_LLM_MODEL_DIR/$DIRECTOR_LLM_MODEL_FILE}"
DIRECTOR_LLM_CTX="${DIRECTOR_LLM_CTX:-2048}"
DIRECTOR_LLM_N_GPU_LAYERS="${DIRECTOR_LLM_N_GPU_LAYERS:-8}"
DIRECTOR_LLM_THREADS="${DIRECTOR_LLM_THREADS:-$(nproc)}"
DIRECTOR_LLM_BATCH="${DIRECTOR_LLM_BATCH:-512}"
DIRECTOR_LLM_UBATCH="${DIRECTOR_LLM_UBATCH:-512}"
DIRECTOR_LLM_FLASH_ATTN="${DIRECTOR_LLM_FLASH_ATTN:-on}"
DIRECTOR_LLM_REASONING="${DIRECTOR_LLM_REASONING:-off}"
DIRECTOR_LLM_REASONING_FORMAT="${DIRECTOR_LLM_REASONING_FORMAT:-none}"
DIRECTOR_LLM_REASONING_BUDGET="${DIRECTOR_LLM_REASONING_BUDGET:-0}"
DIRECTOR_LLM_DAEMON="${DIRECTOR_LLM_DAEMON:-0}"
DIRECTOR_LLM_PID_FILE="${DIRECTOR_LLM_PID_FILE:-$ROOT_DIR/status/director_llm_server.pid}"
DIRECTOR_LLM_LOG_FILE="${DIRECTOR_LLM_LOG_FILE:-$ROOT_DIR/status/director_llm_server.log}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$ROOT_DIR/tools/llama.cpp/build/bin/llama-server}"

mkdir -p "$ROOT_DIR/status" "$DIRECTOR_LLM_MODEL_DIR"

if [ ! -x "$ENSURE_LLAMA_CPP" ]; then
  echo "[director-llm] ERROR: missing helper $ENSURE_LLAMA_CPP" >&2
  exit 1
fi

if [ ! -f "$DIRECTOR_LLM_MODEL_PATH" ]; then
  if [ ! -x "$DOWNLOAD_MODEL" ]; then
    echo "[director-llm] ERROR: missing helper $DOWNLOAD_MODEL" >&2
    exit 1
  fi
  "$DOWNLOAD_MODEL"
fi

"$ENSURE_LLAMA_CPP"

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "[director-llm] ERROR: llama-server missing at $LLAMA_SERVER_BIN" >&2
  exit 1
fi

health_url="${DIRECTOR_LLM_BASE_URL%/}/v1/models"
if curl -fsS "$health_url" >/dev/null 2>&1; then
  echo "[director-llm] server already reachable at $health_url"
  exit 0
fi

server_args=(
  --host "$DIRECTOR_LLM_HOST"
  --port "$DIRECTOR_LLM_PORT"
  -m "$DIRECTOR_LLM_MODEL_PATH"
  -c "$DIRECTOR_LLM_CTX"
  -ngl "$DIRECTOR_LLM_N_GPU_LAYERS"
  -t "$DIRECTOR_LLM_THREADS"
  -b "$DIRECTOR_LLM_BATCH"
  -ub "$DIRECTOR_LLM_UBATCH"
  --jinja
  -rea "$DIRECTOR_LLM_REASONING"
  --reasoning-format "$DIRECTOR_LLM_REASONING_FORMAT"
  --reasoning-budget "$DIRECTOR_LLM_REASONING_BUDGET"
)

if [ "$DIRECTOR_LLM_FLASH_ATTN" = "on" ] || [ "$DIRECTOR_LLM_FLASH_ATTN" = "off" ] || [ "$DIRECTOR_LLM_FLASH_ATTN" = "auto" ]; then
  server_args+=(-fa "$DIRECTOR_LLM_FLASH_ATTN")
fi

echo "[director-llm] starting llama-server with model $DIRECTOR_LLM_MODEL_PATH"
if [ "$DIRECTOR_LLM_DAEMON" = "1" ]; then
  : > "$DIRECTOR_LLM_LOG_FILE"
  setsid "$LLAMA_SERVER_BIN" "${server_args[@]}" </dev/null >>"$DIRECTOR_LLM_LOG_FILE" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$DIRECTOR_LLM_PID_FILE"
  for _ in $(seq 1 60); do
    if curl -fsS "$health_url" >/dev/null 2>&1; then
      echo "[director-llm] server ready at $DIRECTOR_LLM_BASE_URL"
      exit 0
    fi
    sleep 2
  done
  echo "[director-llm] ERROR: server did not become ready, see $DIRECTOR_LLM_LOG_FILE" >&2
  exit 1
fi

exec "$LLAMA_SERVER_BIN" "${server_args[@]}"
