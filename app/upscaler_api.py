from __future__ import annotations

import os
import re
import uuid
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


EDIT_ROOT = Path(os.getenv("EDIT_ROOT", "/workspace"))
EXPORT_DIR = EDIT_ROOT / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Preferred AI entrypoint (PyTorch CUDA, no Vulkan dependency)
AI_SCRIPT_CANDIDATES: List[Path] = [
    Path("/workspace/realesrgan_gpu_pack/upscale_video_ai_cuda.sh"),
    Path("/workspace/upscale_video_ai_cuda.sh"),
]
AI_REPO_DIR = Path("/workspace/tools/realesrgan_ai/Real-ESRGAN")
AI_VENV_DIR = Path("/workspace/tools/realesrgan_ai/venv")


class UpscaleVideoRequest(BaseModel):
    # Required input video path (absolute or workspace-relative)
    input_path: Optional[str] = None
    in_arg: Optional[str] = Field(default=None, alias="in")

    # Output handling
    output_name: Optional[str] = None
    output_path: Optional[str] = Field(default=None, alias="out")

    # Quality/speed knobs
    scale: Optional[int] = 2
    model: Optional[str] = "x2"  # accepts x2|x4|anime and legacy names
    target: Optional[str] = "none"
    tile: Optional[int] = 0
    keep_frames: Optional[bool] = False

    # API smoke/debug mode for n8n testing without running long jobs
    dry_run: Optional[bool] = False

    model_config = {"populate_by_name": True}


def _sanitize_name(name: str) -> str:
    n = re.sub(r"[^\w\-. ]+", "_", name).strip()
    if not n.lower().endswith(".mp4"):
        n += ".mp4"
    return n


def _resolve_input(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (EDIT_ROOT / p).resolve()
    return p


def _pick_input(req: UpscaleVideoRequest) -> Optional[str]:
    return req.input_path or req.in_arg


def _resolve_output(req: UpscaleVideoRequest) -> Path:
    if req.output_path:
        p = Path(req.output_path)
        if not p.is_absolute():
            p = (EDIT_ROOT / p).resolve()
        if p.suffix.lower() != ".mp4":
            p = p.with_suffix(".mp4")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    base_name = req.output_name or f"up_ai_{uuid.uuid4().hex[:8]}.mp4"
    out_name = _sanitize_name(base_name)
    return (EXPORT_DIR / out_name).resolve()


def _pick_ai_script() -> Optional[Path]:
    for p in AI_SCRIPT_CANDIDATES:
        if p.exists():
            return p
    return None


def _normalize_model(model: Optional[str], scale: Optional[int]) -> str:
    m = (model or "").strip().lower()
    if m in {"x2", "realesrgan_x2plus", "realesrgan-x2plus", "x2plus"}:
        return "x2"
    if m in {"x4", "realesrgan_x4plus", "realesrgan-x4plus", "x4plus"}:
        return "x4"
    if "anime" in m:
        return "anime"
    if scale == 4:
        return "x4"
    return "x2"


def _tail(text: str, max_lines: int = 120) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _probe_video(path: Path) -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,avg_frame_rate,codec_type",
                "-of",
                "default=nw=1",
                str(path),
            ],
            text=True,
        )
        return {"ok": True, "probe": out}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": str(e)}


def upscale_video(req: UpscaleVideoRequest) -> Dict[str, Any]:
    script = _pick_ai_script()
    if not script:
        return {
            "ok": False,
            "error": "AI script not found",
            "expected_paths": [str(p) for p in AI_SCRIPT_CANDIDATES],
        }

    req_input = _pick_input(req)
    if not req_input:
        return {"ok": False, "error": "input_path (or alias 'in') is required"}

    in_path = _resolve_input(req_input)
    if not in_path.exists():
        return {"ok": False, "error": f"input not found: {in_path}"}

    out_path = _resolve_output(req)
    model_norm = _normalize_model(req.model, req.scale)
    target = (req.target or "none").strip().lower()
    tile = int(req.tile or 0)

    if target != "none" and not re.match(r"^\d+x\d+$", target):
        return {"ok": False, "error": f"invalid target: {target}. use WxH or none"}

    cmd = [
        "bash",
        str(script),
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--model",
        model_norm,
        "--target",
        target,
        "--tile",
        str(tile),
    ]
    if req.keep_frames:
        cmd.append("--keep-frames")

    if req.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "backend": "realesrgan_ai_cuda",
            "script": str(script),
            "command": cmd,
            "resolved_input": str(in_path),
            "resolved_output": str(out_path),
            "model": model_norm,
            "target": target,
            "tile": tile,
        }

    if not AI_REPO_DIR.exists() or not AI_VENV_DIR.exists():
        return {
            "ok": False,
            "error": "AI runtime not installed. Run: bash /workspace/realesrgan_gpu_pack/setup_ai_cuda.sh",
            "expected_repo": str(AI_REPO_DIR),
            "expected_venv": str(AI_VENV_DIR),
            "script": str(script),
            "command": cmd,
        }

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "upscale command failed",
            "returncode": proc.returncode,
            "backend": "realesrgan_ai_cuda",
            "script": str(script),
            "command": cmd,
            "log_tail": _tail(proc.stdout),
        }

    if not out_path.exists():
        return {
            "ok": False,
            "error": f"output not created: {out_path}",
            "backend": "realesrgan_ai_cuda",
            "log_tail": _tail(proc.stdout),
        }

    rel_out = os.path.relpath(out_path, EDIT_ROOT)
    probe = _probe_video(out_path)

    return {
        "ok": True,
        "backend": "realesrgan_ai_cuda",
        "output_path": rel_out,
        "output_name": out_path.name,
        "output_abs": str(out_path),
        "model": model_norm,
        "target": target,
        "tile": tile,
        "command": cmd,
        "log_tail": _tail(proc.stdout, max_lines=80),
        "probe": probe,
    }
