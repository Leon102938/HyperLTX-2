from __future__ import annotations

import os
import re
import json
import time
import uuid
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

from pydantic import BaseModel, Field


EDIT_ROOT = Path(os.getenv("EDIT_ROOT", "/workspace"))
EXPORT_DIR = EDIT_ROOT / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Preferred AI entrypoint (PyTorch CUDA, no Vulkan dependency)
AI_SCRIPT_CANDIDATES: List[Path] = [
    Path("/workspace/upscaler_installer_minimal/upscale_video_ai_cuda.sh"),
    Path("/workspace/realesrgan_gpu_pack/upscale_video_ai_cuda.sh"),
    Path("/workspace/upscale_video_ai_cuda.sh"),
]
AI_REPO_DIR = Path("/workspace/tools/realesrgan_ai/Real-ESRGAN")
AI_VENV_DIR = Path("/workspace/tools/realesrgan_ai/venv")
UPSCALE_JOBS_DIR = Path("/workspace/jobs/upscale")
UPSCALE_JOBS_DIR.mkdir(parents=True, exist_ok=True)
_UPSCALE_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("UPSCALER_JOB_WORKERS", "1")))
_UPSCALE_JOBS: Dict[str, Dict[str, Any]] = {}
_UPSCALE_LOCK = threading.Lock()
_PROGRESS_RE = re.compile(r"Testing\s+(\d+)\s+frame_")


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


class UpscaleSubmitRequest(UpscaleVideoRequest):
    # Optional remote source. If set and no input_path is provided, API downloads into the job folder.
    input_url: Optional[str] = None


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


def _probe_frame_count(path: Path) -> int:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def _read_log_tail(log_file: Path, tail: int = 120) -> str:
    if not log_file.exists():
        return ""
    max_lines = min(max(int(tail), 1), 1000)
    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=max_lines))


def _prepare_upscale(req: UpscaleVideoRequest) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    script = _pick_ai_script()
    if not script:
        return None, {
            "ok": False,
            "error": "AI script not found",
            "expected_paths": [str(p) for p in AI_SCRIPT_CANDIDATES],
        }

    req_input = _pick_input(req)
    if not req_input:
        return None, {"ok": False, "error": "input_path (or alias 'in') is required"}

    in_path = _resolve_input(req_input)
    if not in_path.exists():
        return None, {"ok": False, "error": f"input not found: {in_path}"}

    out_path = _resolve_output(req)
    model_norm = _normalize_model(req.model, req.scale)
    target = (req.target or "none").strip().lower()
    tile = int(req.tile or 0)

    if target != "none" and not re.match(r"^\d+x\d+$", target):
        return None, {"ok": False, "error": f"invalid target: {target}. use WxH or none"}

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

    if not AI_REPO_DIR.exists() or not AI_VENV_DIR.exists():
        return None, {
            "ok": False,
            "error": "AI runtime not installed. Run: bash /workspace/upscaler_installer_minimal/install_realesrgan_ai_pod.sh",
            "expected_repo": str(AI_REPO_DIR),
            "expected_venv": str(AI_VENV_DIR),
            "script": str(script),
            "command": cmd,
        }

    return {
        "script": script,
        "in_path": in_path,
        "out_path": out_path,
        "model_norm": model_norm,
        "target": target,
        "tile": tile,
        "cmd": cmd,
    }, None


def _job_file(job_id: str) -> Path:
    return UPSCALE_JOBS_DIR / job_id / "job_status.json"


def _persist_job(job_id: str, data: Dict[str, Any]) -> None:
    jf = _job_file(job_id)
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _public_video_url(path_str: str) -> str:
    p = Path(path_str).resolve()
    jobs_root = Path("/workspace/jobs").resolve()
    exports_root = Path("/workspace/exports").resolve()
    try:
        rel = p.relative_to(jobs_root)
        return f"/jobs/{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = p.relative_to(exports_root)
        return f"/exports/{rel.as_posix()}"
    except ValueError:
        pass
    return str(p)


def _download_input_for_job(job_id: str, input_url: str) -> Path:
    parsed = urlparse(input_url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm"}:
        ext = ".mp4"
    job_dir = UPSCALE_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{ext}"
    urlretrieve(input_url, str(input_path))
    return input_path


def _build_job_request(job_id: str, req: UpscaleSubmitRequest) -> UpscaleVideoRequest:
    payload = req.model_dump()
    if not payload.get("input_path") and not payload.get("in_arg") and payload.get("input_url"):
        downloaded = _download_input_for_job(job_id, str(payload["input_url"]))
        payload["input_path"] = str(downloaded)
    if not payload.get("output_path") and not payload.get("output_name"):
        payload["output_path"] = str((UPSCALE_JOBS_DIR / job_id / f"{job_id}.mp4").resolve())
    payload["dry_run"] = False
    payload.pop("input_url", None)
    return UpscaleVideoRequest(**payload)


def _run_submit_job(job_id: str, req: UpscaleSubmitRequest) -> None:
    now = time.time()
    job_dir = UPSCALE_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_file = job_dir / "job.log"
    with _UPSCALE_LOCK:
        job = _UPSCALE_JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = now
        job["log_file"] = str(log_file)
        job["log_url"] = f"/upscale/log/{job_id}"
        _persist_job(job_id, job)

    try:
        run_req = _build_job_request(job_id, req)
        prepared, prep_err = _prepare_upscale(run_req)
        if prep_err:
            raise RuntimeError(prep_err.get("error", "upscale prepare failed"))

        assert prepared is not None
        in_path = Path(prepared["in_path"])
        out_path = Path(prepared["out_path"])
        cmd = list(prepared["cmd"])
        model_norm = str(prepared["model_norm"])
        target = str(prepared["target"])
        tile = int(prepared["tile"])

        total_frames = _probe_frame_count(in_path)
        with _UPSCALE_LOCK:
            job = _UPSCALE_JOBS[job_id]
            job["command"] = cmd
            job["input_path"] = str(in_path)
            job["output_abs"] = str(out_path)
            job["progress"] = {"done": 0, "total": total_frames}
            _persist_job(job_id, job)

        tail_lines: List[str] = []
        last_persist = time.time()
        return_code = 0

        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                stripped = line.rstrip("\n")
                tail_lines.append(stripped)
                if len(tail_lines) > 220:
                    tail_lines.pop(0)
                m = _PROGRESS_RE.search(stripped)
                if m:
                    done = int(m.group(1)) + 1
                    with _UPSCALE_LOCK:
                        job = _UPSCALE_JOBS[job_id]
                        job["progress"] = {"done": done, "total": total_frames}
                if time.time() - last_persist > 1.5:
                    with _UPSCALE_LOCK:
                        job = _UPSCALE_JOBS[job_id]
                        job["log_tail"] = "\n".join(tail_lines[-80:])
                        _persist_job(job_id, job)
                    last_persist = time.time()
            return_code = proc.wait()

        if return_code != 0:
            result = {
                "ok": False,
                "error": "upscale command failed",
                "returncode": return_code,
                "backend": "realesrgan_ai_cuda",
                "command": cmd,
                "log_tail": "\n".join(tail_lines[-120:]),
            }
        elif not out_path.exists():
            result = {
                "ok": False,
                "error": f"output not created: {out_path}",
                "backend": "realesrgan_ai_cuda",
                "command": cmd,
                "log_tail": "\n".join(tail_lines[-120:]),
            }
        else:
            rel_out = os.path.relpath(out_path, EDIT_ROOT)
            result = {
                "ok": True,
                "backend": "realesrgan_ai_cuda",
                "output_path": rel_out,
                "output_name": out_path.name,
                "output_abs": str(out_path),
                "model": model_norm,
                "target": target,
                "tile": tile,
                "command": cmd,
                "input_probe": _probe_video(in_path),
                "output_probe": _probe_video(out_path),
                "log_tail": "\n".join(tail_lines[-120:]),
            }

        done_at = time.time()
        with _UPSCALE_LOCK:
            job = _UPSCALE_JOBS[job_id]
            job["finished_at"] = done_at
            job["result"] = result
            if result.get("ok"):
                output_abs = str(result.get("output_abs", ""))
                job["status"] = "succeeded"
                job["ok"] = True
                job["video_url"] = _public_video_url(output_abs)
                job["output_path"] = result.get("output_path")
                job["output_name"] = result.get("output_name")
                job["progress"] = {"done": total_frames or job.get("progress", {}).get("done", 0), "total": total_frames}
            else:
                job["status"] = "failed"
                job["ok"] = False
                job["error"] = result.get("error", "upscale failed")
            job["log_tail"] = result.get("log_tail", "")
            _persist_job(job_id, job)
    except Exception as e:
        done_at = time.time()
        with _UPSCALE_LOCK:
            job = _UPSCALE_JOBS[job_id]
            job["status"] = "failed"
            job["ok"] = False
            job["error"] = str(e)
            job["finished_at"] = done_at
            if log_file.exists():
                job["log_tail"] = _read_log_tail(log_file, 120)
            _persist_job(job_id, job)


def submit_upscale_job(req: UpscaleSubmitRequest) -> Dict[str, Any]:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "job_id": jid,
        "ok": False,
        "status": "queued",
        "error": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "video_url": None,
        "output_path": None,
        "output_name": None,
        "input_path": None,
        "output_abs": None,
        "command": None,
        "progress": {"done": 0, "total": 0},
        "log_file": str((UPSCALE_JOBS_DIR / jid / "job.log").resolve()),
        "log_url": f"/upscale/log/{jid}",
        "log_tail": "",
        "result": None,
    }
    with _UPSCALE_LOCK:
        _UPSCALE_JOBS[jid] = job
        _persist_job(jid, job)
    _UPSCALE_EXECUTOR.submit(_run_submit_job, jid, req)
    return {
        "job_id": jid,
        "status": "queued",
        "status_url": f"/upscale/get/{jid}",
        "get_url": f"/upscale/get/{jid}",
        "log_url": f"/upscale/log/{jid}",
    }


def get_upscale_job(job_id: str) -> Dict[str, Any]:
    with _UPSCALE_LOCK:
        job = _UPSCALE_JOBS.get(job_id)
    if job is None:
        jf = _job_file(job_id)
        if not jf.exists():
            return {"ok": False, "status": "not_found", "job_id": job_id, "error": "job not found"}
        try:
            job = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": False, "status": "not_found", "job_id": job_id, "error": "job status unreadable"}

    if job.get("status") == "succeeded":
        return {
            "ok": True,
            "status": "succeeded",
            "job_id": job_id,
            "video_url": job.get("video_url"),
            "output_path": job.get("output_path"),
            "output_name": job.get("output_name"),
            "progress": job.get("progress", {"done": 0, "total": 0}),
            "log_url": job.get("log_url", f"/upscale/log/{job_id}"),
            "command": job.get("command"),
            "input_path": job.get("input_path"),
            "result": job.get("result"),
        }

    return {
        "ok": False,
        "status": job.get("status", "unknown"),
        "job_id": job_id,
        "error": job.get("error"),
        "progress": job.get("progress", {"done": 0, "total": 0}),
        "log_url": job.get("log_url", f"/upscale/log/{job_id}"),
        "command": job.get("command"),
        "input_path": job.get("input_path"),
    }


def get_upscale_job_log(job_id: str, tail: int = 120) -> Dict[str, Any]:
    with _UPSCALE_LOCK:
        job = _UPSCALE_JOBS.get(job_id)

    if job is None:
        jf = _job_file(job_id)
        if not jf.exists():
            return {"ok": False, "status": "not_found", "job_id": job_id, "error": "job not found"}
        try:
            job = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": False, "status": "not_found", "job_id": job_id, "error": "job status unreadable"}

    log_path = Path(job.get("log_file") or str((UPSCALE_JOBS_DIR / job_id / "job.log").resolve()))
    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "log_file": str(log_path),
        "log_tail": _read_log_tail(log_path, tail=tail),
    }


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
            "error": "AI runtime not installed. Run: bash /workspace/upscaler_installer_minimal/install_realesrgan_ai_pod.sh",
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
