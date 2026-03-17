import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
STATUS_DIR = Path("/workspace/status")
EXPORT_DIR = Path("/workspace/exports")
JOBS_ROOT = Path("/workspace/jobs/qwen_tts")
MODEL_PATH = Path("/workspace/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TOKENIZER_PATH = Path("/workspace/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz")
ENV_FLAG = STATUS_DIR / "qwen_tts_env_ready"
TOKENIZER_FLAG = STATUS_DIR / "qwen_tts_tokenizer_ready"
MODEL_FLAG = STATUS_DIR / "qwen_tts_model_ready"
QWEN_PYTHON = Path("/workspace/venvs/qwen3-tts/bin/python")

WORKER_CODE = r"""
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

job_json = Path(os.environ["QWEN_JOB_JSON"])
with open(job_json, "r", encoding="utf-8") as f:
    req = json.load(f)

model_path = req["model_path"]
output_path = req["output_path"]

kwargs = {
    "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
}
if torch.cuda.is_available():
    kwargs["attn_implementation"] = "flash_attention_2"

try:
    model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
except TypeError:
    kwargs.pop("attn_implementation", None)
    model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)

with torch.inference_mode():
    wavs, sample_rate = model.generate_custom_voice(
        text=req["text"],
        language=req["language"],
        speaker=req["speaker"],
        instruct=req.get("instruct") or "",
    )

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
sf.write(output_path, wavs[0], sample_rate, subtype="PCM_16")

result = {
    "sample_rate": int(sample_rate),
    "output_path": output_path,
}
with open(os.environ["QWEN_RESULT_JSON"], "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
"""


class QwenCustomVoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("German", min_length=1)
    speaker: str = Field("Ryan", min_length=1)
    instruct: Optional[str] = Field(
        "A deep male German-speaking voice, slightly faster pace, a bit more energetic and lively, confident, warm, natural, realistic, premium YouTube voiceover style, clear pronunciation, engaging but not overexcited."
    )
    output_name: Optional[str] = None
    job_id: Optional[str] = None


@dataclass
class QwenTTSJob:
    id: str
    status: str
    state: str
    ts: float
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    output_name: str = ""
    output_path: str = ""
    file_url: str = ""
    log_file: str = ""
    text: str = ""
    language: str = ""
    speaker: str = ""
    instruct: Optional[str] = None
    sample_rate: Optional[int] = None


def _ready_payload() -> dict:
    env_ready = ENV_FLAG.exists() and QWEN_PYTHON.exists()
    tokenizer_ready = TOKENIZER_FLAG.exists() and TOKENIZER_PATH.exists()
    model_ready = MODEL_FLAG.exists() and MODEL_PATH.exists()
    return {
        "ready": env_ready and tokenizer_ready and model_ready,
        "env_ready": env_ready,
        "tokenizer_ready": tokenizer_ready,
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "python_path": str(QWEN_PYTHON),
    }


def _sanitize_output_name(name: Optional[str]) -> str:
    if not name:
        return f"qwen_{uuid.uuid4().hex[:12]}.wav"

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    if not cleaned:
        cleaned = f"qwen_{uuid.uuid4().hex[:12]}"
    if not cleaned.endswith(".wav"):
        cleaned = f"{cleaned}.wav"
    return cleaned


class _QwenTTSService:
    def __init__(self):
        self.jobs_root = JOBS_ROOT.resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, QwenTTSJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _status_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job_status.json"

    def _request_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "request.json"

    def _result_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "result.json"

    def _persist(self, job: QwenTTSJob) -> None:
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._status_file(job.id).write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    def _read_persisted(self, job_id: str) -> Optional[dict]:
        status_file = self._status_file(job_id)
        if not status_file.exists():
            return None
        return json.loads(status_file.read_text(encoding="utf-8"))

    async def create_job(self, req: QwenCustomVoiceRequest) -> QwenTTSJob:
        ready = _ready_payload()
        if not ready["ready"]:
            raise HTTPException(status_code=503, detail=ready)

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        job_id = req.job_id or f"qwen_{uuid.uuid4().hex[:12]}"
        output_name = _sanitize_output_name(req.output_name)
        output_path = EXPORT_DIR / output_name
        log_file = self._job_dir(job_id) / "job.log"
        file_url = f"{BASE_URL}/exports/{output_name}" if BASE_URL else f"/exports/{output_name}"
        now = time.time()

        job = QwenTTSJob(
            id=job_id,
            status="queued",
            state="queued",
            ts=now,
            created_at=now,
            output_name=output_name,
            output_path=str(output_path),
            file_url=file_url,
            log_file=str(log_file),
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
        )
        self.jobs[job_id] = job
        self._persist(job)
        self._request_file(job_id).write_text(
            json.dumps(
                {
                    "text": req.text,
                    "language": req.language,
                    "speaker": req.speaker,
                    "instruct": req.instruct,
                    "output_path": str(output_path),
                    "model_path": str(MODEL_PATH),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        await self._queue.put(job_id)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
        return job

    async def _worker_loop(self) -> None:
        while not self._queue.empty():
            job_id = await self._queue.get()
            async with self._lock:
                job = self.jobs[job_id]
                job.status = job.state = "running"
                job.started_at = job.ts = time.time()
                self._persist(job)

                job_dir = self._job_dir(job.id)
                log_path = Path(job.log_file)
                result_path = self._result_file(job.id)
                request_path = self._request_file(job.id)
                if result_path.exists():
                    result_path.unlink()

                try:
                    with open(log_path, "w", encoding="utf-8") as log_file:
                        log_file.write("backend: qwen3-tts\n")
                        log_file.write(f"job_id: {job.id}\n")
                        log_file.write(f"python: {QWEN_PYTHON}\n")
                        log_file.write(f"speaker: {job.speaker}\n")
                        log_file.write(f"language: {job.language}\n")
                        log_file.write(f"output_path: {job.output_path}\n\n")
                        log_file.flush()

                        env = os.environ.copy()
                        env["QWEN_JOB_JSON"] = str(request_path)
                        env["QWEN_RESULT_JSON"] = str(result_path)
                        env["PYTHONNOUSERSITE"] = "1"

                        proc = await asyncio.create_subprocess_exec(
                            str(QWEN_PYTHON),
                            "-c",
                            WORKER_CODE,
                            cwd="/workspace",
                            env=env,
                            stdout=log_file,
                            stderr=log_file,
                        )
                        rc = await proc.wait()
                        job.exit_code = rc

                    if rc != 0:
                        raise RuntimeError(f"qwen worker exited with code {rc}")
                    if not result_path.exists():
                        raise RuntimeError("qwen worker did not write result.json")

                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    job.sample_rate = result.get("sample_rate")
                    job.status = job.state = "succeeded"
                    job.error = None
                except Exception as exc:
                    job.status = job.state = "failed"
                    job.error = str(exc)
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        lines = [line.strip() for line in reversed(log_text.splitlines()) if line.strip()]
                        if lines:
                            job.error = lines[-1] if "Traceback" in lines[-1] else lines[0]
                job.finished_at = job.ts = time.time()
                self._persist(job)
            self._queue.task_done()

    def get_status(self, job_id: str) -> dict:
        if job_id in self.jobs:
            return asdict(self.jobs[job_id])

        persisted = self._read_persisted(job_id)
        if persisted is None:
            raise HTTPException(status_code=404, detail="job_id not found")
        return persisted

    def get_result(self, job_id: str) -> dict:
        info = self.get_status(job_id)

        if info.get("status") == "succeeded":
            return {
                "ok": True,
                "job_id": job_id,
                "status": "succeeded",
                "state": "succeeded",
                "speaker": info.get("speaker"),
                "language": info.get("language"),
                "sample_rate": info.get("sample_rate"),
                "output_name": info.get("output_name"),
                "output_path": info.get("output_path"),
                "file_url": info.get("file_url"),
            }

        return {
            "ok": False,
            "job_id": job_id,
            "status": info.get("status"),
            "state": info.get("state"),
            "error": info.get("error"),
        }


_service = _QwenTTSService()


@router.post("/custom_voice")
async def qwen_tts_custom_voice(req: QwenCustomVoiceRequest):
    job = await _service.create_job(req)
    return {
        "job_id": job.id,
        "state": job.state,
        "status_url": f"/qwen_tts/status/{job.id}",
        "get_url": f"/qwen_tts/get/{job.id}",
    }


@router.get("/status/{job_id}")
def qwen_tts_status(job_id: str):
    return _service.get_status(job_id)


@router.get("/get/{job_id}")
def qwen_tts_get(job_id: str):
    return _service.get_result(job_id)
