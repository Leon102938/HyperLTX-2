import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


router = APIRouter()

HIDREAM_PY = sys.executable
HF_HOME = os.environ.get("HF_HOME", "/workspace/.cache/hf")
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
STATUS_DIR = Path(os.environ.get("STATUS_DIR", "/workspace/status"))
HIDREAM_READY_FLAG = Path(os.environ.get("HIDREAM_READY_FLAG", str(STATUS_DIR / "hidream_ready")))
HIDREAM_MODEL_ID = os.environ.get("HIDREAM_MODEL_ID", "HiDream-ai/HiDream-O1-Dev")
JOBS_ROOT = Path(os.environ.get("HIDREAM_JOBS_ROOT", "/workspace/jobs/hidream"))


class HiDreamJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = Field("HiDream-O1-Dev", min_length=1)
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(768, ge=256, le=2048)
    steps: int = Field(28, ge=1, le=60)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(42, ge=0)
    job_id: Optional[str] = None


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _write_status(job_id: str, state: str, extra: Optional[Dict[str, Any]] = None) -> None:
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "state": state, "ts": time.time()}
    if extra:
        payload.update(extra)
    (d / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_status(job_id: str) -> Dict[str, Any]:
    path = _job_dir(job_id) / "status.json"
    if not path.exists():
        raise FileNotFoundError
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("state") == "succeeded":
        data["file_url"] = f"{BASE_URL}/hidream/jobs/{job_id}/file"
    return data


async def _run_job(job_id: str) -> None:
    d = _job_dir(job_id)
    req_path = d / "request.json"
    out_path = d / "out.png"
    stdout_path = d / "stdout.log"
    stderr_path = d / "stderr.log"

    _write_status(job_id, "running")

    worker = r"""
import os, json, torch, sys
try:
    from diffusers import AutoPipelineForText2Image

    with open(os.environ["JOB_JSON"], "r", encoding="utf-8") as f:
        req = json.load(f)

    prompt = req["prompt"]
    width = int(req["width"])
    height = int(req["height"])
    steps = int(req["steps"])
    guidance = float(req["guidance_scale"])
    seed = req.get("seed", None)
    out_path = req["out_path"]
    model_id = os.environ.get("HIDREAM_MODEL_ID", "HiDream-ai/HiDream-O1-Dev")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    print(f"Loading {model_id}...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    gen = None
    if seed is not None:
        gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    print(f"Generating HiDream image for: {prompt}")
    img = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    ).images[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"Saved image to {out_path}.")
except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)
"""

    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["JOB_JSON"] = str(req_path)
    env["HIDREAM_MODEL_ID"] = HIDREAM_MODEL_ID
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + str(Path.cwd())

    try:
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = await asyncio.create_subprocess_exec(
                HIDREAM_PY,
                "-c",
                worker,
                stdout=out_f,
                stderr=err_f,
                env=env,
            )
            return_code = await proc.wait()

        if return_code != 0:
            _write_status(job_id, "failed", extra={"error": f"process exited with code {return_code}"})
        elif not out_path.exists():
            _write_status(job_id, "failed", extra={"error": "image file was not created"})
        else:
            _write_status(job_id, "succeeded", extra={"output_path": str(out_path)})
    except Exception as exc:
        _write_status(job_id, "failed", extra={"error": f"API could not start job: {exc}"})


@router.get("/ready")
def hidream_ready():
    ok = HIDREAM_READY_FLAG.exists()
    return {"ready": ok, "flag": str(HIDREAM_READY_FLAG), "model": "HiDream-O1-Dev", "steps": 28, "message": "HiDream-O1-Dev bereit." if ok else "HiDream-O1-Dev nicht bereit."}


@router.post("/jobs")
async def hidream_submit(req: HiDreamJobRequest):
    if not HIDREAM_READY_FLAG.exists():
        raise HTTPException(status_code=503, detail="HiDream-O1-Dev not ready (flag missing).")

    job_id = req.job_id or f"hidream_{uuid.uuid4().hex[:12]}"
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    request_payload = req.model_dump()
    request_payload["steps"] = 28
    request_payload["out_path"] = str(d / "out.png")
    (d / "request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_status(job_id, "queued")
    asyncio.create_task(_run_job(job_id))

    return {
        "job_id": job_id,
        "status_url": f"{BASE_URL}/hidream/jobs/{job_id}",
        "state": "queued",
    }


@router.get("/jobs/{job_id}")
def hidream_status(job_id: str):
    try:
        return _read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job_id not found")


@router.get("/jobs/{job_id}/file")
def hidream_file(job_id: str):
    d = _job_dir(job_id)
    out_path = d / "out.png"

    if not out_path.exists():
        try:
            status = _read_status(job_id)
            if status.get("state") in ("queued", "running"):
                raise HTTPException(status_code=409, detail=f"not ready yet: {status.get('state')}")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="job_id not found")
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(str(out_path), media_type="image/png", filename=f"{job_id}.png")
