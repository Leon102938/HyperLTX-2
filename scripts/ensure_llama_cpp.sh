#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/tools/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"
LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-$LLAMA_CPP_DIR/build/bin/llama-cli}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-$LLAMA_CPP_DIR/build/bin}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_BUILD_JOBS="${LLAMA_CPP_BUILD_JOBS:-$(nproc)}"
LLAMA_CPP_FALLBACK_SRC_DIR="${LLAMA_CPP_FALLBACK_SRC_DIR:-$ROOT_DIR/tools/llama.cpp.upstream}"

runtime_ready() {
  if [ ! -x "$LLAMA_SERVER_BIN" ] || [ ! -x "$LLAMA_CLI_BIN" ]; then
    return 1
  fi
  if ldd "$LLAMA_SERVER_BIN" 2>/dev/null | grep -q 'not found'; then
    return 1
  fi
  return 0
}

repair_runtime() {
  local lib_name
  local versioned_path
  local versioned_file

  if [ ! -d "$LLAMA_BIN_DIR" ]; then
    return 1
  fi

  chmod +x "$LLAMA_SERVER_BIN" "$LLAMA_CLI_BIN" 2>/dev/null || true

  for lib_name in \
    libggml-base \
    libggml-cpu \
    libggml-cuda \
    libggml \
    libllama-common \
    libllama \
    libmtmd
  do
    versioned_path="$(find "$LLAMA_BIN_DIR" -maxdepth 1 -type f -name "$lib_name.so.*" | sort -V | tail -n 1)"
    if [ -n "$versioned_path" ]; then
      versioned_file="$(basename "$versioned_path")"
      ln -sfn "$versioned_file" "$LLAMA_BIN_DIR/$lib_name.so.0"
      ln -sfn "$lib_name.so.0" "$LLAMA_BIN_DIR/$lib_name.so"
    fi
  done
}

if runtime_ready; then
  echo "[director-llm] llama.cpp already available at $LLAMA_SERVER_BIN"
  exit 0
fi

repair_runtime
if runtime_ready; then
  echo "[director-llm] repaired existing llama.cpp runtime at $LLAMA_BIN_DIR"
  exit 0
fi

mkdir -p "$(dirname "$LLAMA_CPP_DIR")"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "[director-llm] cloning llama.cpp into $LLAMA_CPP_DIR"
  git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
elif [ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
  echo "[director-llm] ERROR: existing $LLAMA_CPP_DIR is not a llama.cpp source tree" >&2
  exit 1
fi

LLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR"
if [ ! -f "$LLAMA_CPP_DIR/tools/mtmd/models/models.h" ]; then
  echo "[director-llm] existing llama.cpp tree is incomplete; using fresh upstream clone for build source"
  if [ ! -d "$LLAMA_CPP_FALLBACK_SRC_DIR" ]; then
    git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_FALLBACK_SRC_DIR"
  elif [ ! -f "$LLAMA_CPP_FALLBACK_SRC_DIR/CMakeLists.txt" ]; then
    echo "[director-llm] ERROR: fallback source tree $LLAMA_CPP_FALLBACK_SRC_DIR is invalid" >&2
    exit 1
  fi
  LLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_FALLBACK_SRC_DIR"
  rm -rf "$LLAMA_CPP_DIR/build"
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
cmake -S "$LLAMA_CPP_SOURCE_DIR" -B "$LLAMA_CPP_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
echo "[director-llm] building llama-server and llama-cli"
cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server llama-cli -j "$LLAMA_CPP_BUILD_JOBS"
echo "[director-llm] llama.cpp ready"
