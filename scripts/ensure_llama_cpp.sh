#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/tools/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"
LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-$LLAMA_CPP_DIR/build/bin/llama-cli}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_BUILD_JOBS="${LLAMA_CPP_BUILD_JOBS:-$(nproc)}"

if [ -x "$LLAMA_SERVER_BIN" ] && [ -x "$LLAMA_CLI_BIN" ]; then
  echo "[director-llm] llama.cpp already available at $LLAMA_SERVER_BIN"
  exit 0
fi

mkdir -p "$(dirname "$LLAMA_CPP_DIR")"

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  echo "[director-llm] cloning llama.cpp into $LLAMA_CPP_DIR"
  git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[director-llm] installing cmake via pip"
  python -m pip install --no-cache-dir cmake
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "[director-llm] installing ninja via pip"
  python -m pip install --no-cache-dir ninja
fi

echo "[director-llm] configuring llama.cpp build"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
echo "[director-llm] building llama-server and llama-cli"
cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server llama-cli -j "$LLAMA_CPP_BUILD_JOBS"
echo "[director-llm] llama.cpp ready"
