# /workspace/app/LTX2.py
import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# Pfade basierend auf deiner erfolgreichen Test-Struktur
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
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    output_file: str = ""
    log_file: str = ""

class _LTX2Service:
    def __init__(self):
        self.jobs_root = Path(LTX_JOBS_DIR)
        self.jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    async def create_job(self, prompt: str, overrides: Dict[str, Any], job_id: Optional[str]) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = str(job_dir / f"{jid}.mp4")
        log_file = str(job_dir / "job.log")

        job = Job(id=jid, status="queued", created_at=time.time(), output_file=output_file, log_file=log_file)
        self.jobs[jid] = job
        await self._queue.put(jid)
        asyncio.create_task(self._worker_loop()) # Startet Worker falls nicht aktiv
        return jid

    async def _worker_loop(self):
        while not self._queue.empty():
            jid = await self._queue.get()
            async with self._lock:
                await self._run_job(jid)
            self._queue.task_done()

    async def _run_job(self, jid: str):
        job = self.jobs[jid]
        job.status = "running"
        job.started_at = time.time()
        
        # Der Befehl aus deinem Screenshot
        cmd = [
            LTX_PYTHON, "packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py",
            "--checkpoint-path", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-19b-dev-fp8.safetensors",
            "--spatial-upsampler-path", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
            "--distilled-lora", f"{LTX_CKPT_DIR}/ltx-2/ltx-2-19b-distilled-lora-384.safetensors", "1.0",
            "--gemma-root", f"{LTX_CKPT_DIR}/gemma-3",
            "--prompt", job.prompt if hasattr(job, 'prompt') else "Cinematic video", # Prompt Logik
            "--output-path", job.output_file,
            "--enable-fp8"
        ]
        
        # Overrides hinzufügen (z.B. height, width)
        for k, v in (job.overrides if hasattr(job, 'overrides') else {}).items():
            cmd.extend([f"--{k}", str(v)])

        try:
            with open(job.log_file, "w") as lf:
                process = await asyncio.create_subprocess_exec(*cmd, cwd=LTX_ROOT, stdout=lf, stderr=lf)
                await process.wait()
                job.status = "succeeded" if process.returncode == 0 else "failed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
        job.finished_at = time.time()

_service = _LTX2Service()
async def submit_job(req: LTX2JobRequest): return await _service.create_job(req.prompt, req.overrides, req.job_id)
def get_status(job_id: str): return asdict(_service.jobs.get(job_id)) if job_id in _service.jobs else {"error": "not found"}