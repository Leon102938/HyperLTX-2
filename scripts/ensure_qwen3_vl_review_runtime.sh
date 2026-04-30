#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${QWEN3_VL_REVIEW_VENV:-/workspace/venvs/qwen3-vl-review}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[qwen3-vl-runtime] ensure venv: ${VENV_DIR}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

echo "[qwen3-vl-runtime] install isolated review deps"
"${VENV_DIR}/bin/python" -m pip install -U \
  "transformers==5.7.0" \
  "accelerate==1.13.0" \
  "safetensors" \
  "pillow" \
  "qwen-vl-utils==0.0.14" \
  "kernels==0.13.0"

echo "[qwen3-vl-runtime] verify qwen3_vl imports"
"${VENV_DIR}/bin/python" - <<'PY'
import sys
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

print("python", sys.executable)
print("transformers", transformers.__version__, transformers.__file__)
print("torch", torch.__version__, torch.__file__)
print("qwen3vl import ok")
PY

echo "[qwen3-vl-runtime] ready"
