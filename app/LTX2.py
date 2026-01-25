# /workspace/app/LTX2.py
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

LTX_ROOT = "/workspace/LTX-2"
LTX_CKPT_DIR = f"{LTX_ROOT}/checkpoints"
LTX_JOBS_DIR = "/workspace/jobs"
LTX_PYTHON = "python3"

class LTX2JobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None

@dataclass
class Job:
    id: str
    status: str  # queued | running | succeeded | failed
    state: str   # Alias für n8n 'state' Anzeige
    ts: float    # Zeitstempel für n8n
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    output_path: str = "" # Für n8n Anzeige
    output_file: str = "" # Interner Pfad
    log_file: str = ""
    prompt: str = ""
    overrides: Dict[str, Any] = None

class _LTX2Service:
    def __init__(self):
        self.jobs_root = Path(LTX_JOBS_DIR).resolve()
        self.jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def _persist(self, job: Job):
        try:
            jd = Path(self.jobs_root / job.id)
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "job_status.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        except Exception: pass

    async def create_job(self, prompt: str, overrides: Dict[str, Any], job_id: Optional[str]) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        
        job = Job(
            id=jid, status="queued", state="queued", ts=time.time(),
            created_at=time.time(),
            output_path=f"/workspace/jobs/{jid}/{jid}.mp4",
            output_file=str(job_dir / f"{jid}.mp4"),
            log_file=str(job_dir / "job.log"),
            prompt=prompt, overrides=overrides or {}
        )
        self.jobs[jid] = job
        self._persist(job)
        await self._queue.put(jid)
        asyncio.create_task(self._worker_loop())
        return jid

    async def _worker_loop(self):
        while not self._queue.empty():
            jid = await self._queue.get()
            async with self._lock:
                job = self.jobs[jid]
                job.status = job.state = "running"
                job.started_at = time.time()
                self._persist(job)

                # Basis-Kommando für die Two-Stage Pipeline
                cmd = [
                    LTX_PYTHON, "packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py",
                    "--checkpoint-path", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-19b-dev-fp8.safetensors",
                    "--spatial-upsampler-path", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
                    "--distilled-lora", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-19b-distilled-lora-384.safetensors", "0.8",
                    "--gemma-root", f"{LTX_CKPT_DIR}/gemma-3",
                    "--prompt", job.prompt, 
                    "--output-path", job.output_file, 
                    "--enable-fp8"
                ]

                # Dynamische Parameter-Logik
                ov = job.overrides or {}

                # 1. Image-to-Video Logik
                if "img_path" in ov and ov["img_path"]:
                    cmd.extend(["--image", str(ov["img_path"]), "0", "1.0"])

                # 2. LoRA Logik (Unterstützt lora1, lora2, lora3 aus n8n)
                for i in range(1, 4):
                    l_key = f"lora{i}"
                    if l_key in ov and ov[l_key] and isinstance(ov[l_key], list):
                        # Erwartet ["pfad", stärke]
                        cmd.extend(["--lora", str(ov[l_key][0]), str(ov[l_key][1])])

                # 3. Standard-Parameter Mapping (Bindestrich-Fix)
                for k, v in ov.items():
                    if k in ["width", "height", "num_frames", "num_inference_steps", "cfg_guidance_scale", "negative_prompt"]:
                        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
                    elif k == "enhance_prompt" and str(v).lower() in ["true", "an", "1"]:
                        cmd.append("--enhance-prompt")

                try:
                    with open(job.log_file, "w") as lf:
                        proc = await asyncio.create_subprocess_exec(*cmd, cwd=LTX_ROOT, stdout=lf, stderr=lf)
                        rc = await proc.wait()
                        job.status = job.state = "succeeded" if rc == 0 else "failed"
                        job.exit_code = rc
                except Exception as e:
                    job.status = job.state = "failed"
                    job.error = str(e)
                
                job.finished_at = job.ts = time.time()
                self._persist(job)
            self._queue.task_done()

_service = _LTX2Service()
async def submit_job(req: LTX2JobRequest): return await _service.create_job(req.prompt, req.overrides, req.job_id)
def get_status(job_id: str):
    if job_id in _service.jobs: return asdict(_service.jobs[job_id])
    jf = Path(LTX_JOBS_DIR) / job_id / "job_status.json"
    return json.loads(jf.read_text()) if jf.exists() else {"error": "not found"}