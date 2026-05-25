from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


LTX_ROOT = Path("/workspace/LTX-2")
LTX_PYTHON = "python3"
LTX_PYTHONPATH = [
    LTX_ROOT / "packages/ltx-core/src",
    LTX_ROOT / "packages/ltx-pipelines/src",
]

DEFAULT_CHECKPOINT_PATH = LTX_ROOT / "checkpoints/ltx-2.3/ltx-2.3-22b-dev.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_PATH = LTX_ROOT / "checkpoints/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_DISTILLED_LORA_PATH = LTX_ROOT / "checkpoints/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors"
DEFAULT_GEMMA_ROOT = LTX_ROOT / "checkpoints/gemma-3"

GEMMA_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "size": path.stat().st_size if path.exists() and path.is_file() else None}


def _run_help(module: str) -> dict[str, Any]:
    env = os.environ.copy()
    py_path = ":".join(str(path) for path in LTX_PYTHONPATH)
    env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_path
    proc = subprocess.run(
        [LTX_PYTHON, "-m", module, "--help"],
        cwd=str(LTX_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "module": module,
        "command": [LTX_PYTHON, "-m", module, "--help"],
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def ltx_completeness_readiness_report(run_help: bool = True, status_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    gemma_root = DEFAULT_GEMMA_ROOT
    model_shards = sorted(gemma_root.glob("model-*.safetensors")) if gemma_root.exists() else []
    checks: dict[str, Any] = {
        "main_checkpoint": _exists(DEFAULT_CHECKPOINT_PATH),
        "spatial_upsampler": _exists(DEFAULT_SPATIAL_UPSAMPLER_PATH),
        "distilled_lora": _exists(DEFAULT_DISTILLED_LORA_PATH),
        "gemma_root": {"path": str(gemma_root), "exists": gemma_root.exists(), "is_dir": gemma_root.is_dir()},
        "gemma_required_files": {name: _exists(gemma_root / name) for name in GEMMA_REQUIRED_FILES},
        "gemma_model_shards": {"count": len(model_shards), "paths": [str(path) for path in model_shards], "ok": len(model_shards) > 0},
    }
    if run_help:
        checks["a2vid_help"] = _run_help("ltx_pipelines.a2vid_two_stage")
        checks["ti2vid_help"] = _run_help("ltx_pipelines.ti2vid_two_stages")
    else:
        checks["a2vid_help"] = {"ok": None, "skipped": True}
        checks["ti2vid_help"] = {"ok": None, "skipped": True}

    required_ok = [
        checks["main_checkpoint"]["exists"],
        checks["spatial_upsampler"]["exists"],
        checks["distilled_lora"]["exists"],
        checks["gemma_root"]["is_dir"],
        all(item["exists"] for item in checks["gemma_required_files"].values()),
        checks["gemma_model_shards"]["ok"],
        checks["a2vid_help"]["ok"] is True,
        checks["ti2vid_help"]["ok"] is True,
    ]
    status_ready = None if status_payload is None else status_payload.get("ready")
    return {
        "ready": all(required_ok),
        "status_payload_ready": status_ready,
        "status_payload_ready_is_sufficient": status_ready is True and all(required_ok),
        "rule": "ltx_video.ready=null or a generic ready endpoint is not sufficient; checkpoints, Gemma tokenizer/model files, and CLI help must pass.",
        "checks": checks,
    }
