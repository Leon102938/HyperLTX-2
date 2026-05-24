import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
STATUS_DIR = Path("/workspace/status")
JOBS_ROOT = Path("/workspace/jobs/ace_step_1_5")
JOBS_BASE = Path("/workspace/jobs")
ACE_STEP_ROOT = Path("/workspace/ACE-Step-1.5")
ACE_STEP_CHECKPOINTS = ACE_STEP_ROOT / "checkpoints"
ACE_STEP_PYTHON = Path(os.environ.get("ACE_STEP_PYTHON", "/opt/venv/bin/python"))
READY_FLAG = STATUS_DIR / "ace_step_ready"
ENV_FLAG = STATUS_DIR / "ace_step_env_ready"
EXPECTED_MODEL_DIRS = (
    ACE_STEP_CHECKPOINTS / "acestep-v15-turbo",
    ACE_STEP_CHECKPOINTS / "vae",
    ACE_STEP_CHECKPOINTS / "Qwen3-Embedding-0.6B",
    ACE_STEP_CHECKPOINTS / "acestep-5Hz-lm-1.7B",
)

WORKER_CODE = r"""
import json
import os
import sys
from pathlib import Path

job_json = Path(os.environ["ACE_STEP_JOB_JSON"])
result_json = Path(os.environ["ACE_STEP_RESULT_JSON"])

with open(job_json, "r", encoding="utf-8") as f:
    req = json.load(f)

project_root = req["project_root"]
checkpoint_dir = req["checkpoint_dir"]
output_dir = req["output_dir"]
config_path = req["config_path"]
lm_model_path = req["lm_model_path"]
backend = req["backend"]
device = req["device"]

sys.path.insert(0, project_root)
os.chdir(project_root)

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

def sanitize_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): sanitize_jsonable(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [sanitize_jsonable(v) for v in value]
    return str(value)

dit_handler = AceStepHandler()
dit_status, dit_ok = dit_handler.initialize_service(
    project_root=project_root,
    config_path=config_path,
    device=device,
    use_flash_attention=req["use_flash_attention"],
    compile_model=req["compile_model"],
    offload_to_cpu=req["offload_to_cpu"],
    offload_dit_to_cpu=req["offload_dit_to_cpu"],
    quantization=req["quantization"],
    prefer_source=req["prefer_source"],
    use_mlx_dit=req["use_mlx_dit"],
)
if not dit_ok:
    raise RuntimeError(dit_status)

llm_handler = LLMHandler()
llm_status, llm_ok = llm_handler.initialize(
    checkpoint_dir=checkpoint_dir,
    lm_model_path=lm_model_path,
    backend=backend,
    device=device,
    offload_to_cpu=req["lm_offload_to_cpu"],
)
if not llm_ok:
    raise RuntimeError(llm_status)

params = GenerationParams(**req["params"])
config = GenerationConfig(**req["config"])
result = generate_music(
    dit_handler=dit_handler,
    llm_handler=llm_handler,
    params=params,
    config=config,
    save_dir=output_dir,
)

if not result.success:
    raise RuntimeError(result.error or result.status_message or "ACE-Step generation failed")

audios = []
output_paths = []
for audio in result.audios:
    audio_path = audio.get("path") or ""
    if audio_path:
        output_paths.append(audio_path)
    audio_params = audio.get("params")
    if hasattr(audio_params, "to_dict"):
        audio_params = audio_params.to_dict()
    audios.append(
        {
            "path": audio_path,
            "key": audio.get("key"),
            "sample_rate": audio.get("sample_rate"),
            "params": audio_params,
        }
    )

payload = {
    "success": True,
    "status_message": result.status_message,
    "extra_outputs": sanitize_jsonable(
        {
            "time_costs": result.extra_outputs.get("time_costs", {}),
            "lm_metadata": result.extra_outputs.get("lm_metadata", {}),
        }
    ),
    "audios": audios,
    "output_paths": output_paths,
    "primary_output_path": output_paths[0] if output_paths else "",
    "dit_status": dit_status,
    "llm_status": llm_status,
}

with open(result_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
"""


class AceStepGenerateRequest(BaseModel):
    job_id: Optional[str] = None

    config_path: str = "acestep-v15-turbo"
    lm_model_path: str = "acestep-5Hz-lm-1.7B"
    backend: str = "pt"
    device: str = "cuda"
    use_flash_attention: bool = False
    compile_model: bool = False
    offload_to_cpu: bool = False
    offload_dit_to_cpu: bool = False
    lm_offload_to_cpu: bool = False
    quantization: Optional[str] = None
    prefer_source: Optional[str] = None
    use_mlx_dit: bool = True

    task_type: str = "text2music"
    instruction: str = "Fill the audio semantic mask based on the given conditions:"
    reference_audio: Optional[str] = None
    src_audio: Optional[str] = None
    audio_codes: str = ""
    caption: str = ""
    global_caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "unknown"
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0
    enable_normalization: bool = True
    normalization_db: float = -1.0
    fade_in_duration: float = 0.0
    fade_out_duration: float = 0.0
    latent_shift: float = 0.0
    latent_rescale: float = 1.0
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    shift: float = 1.0
    infer_method: str = "ode"
    timesteps: Optional[list[float]] = None
    repainting_start: float = 0.0
    repainting_end: float = -1.0
    chunk_mask_mode: str = "auto"
    repaint_latent_crossfade_frames: int = 10
    repaint_wav_crossfade_sec: float = 0.0
    repaint_mode: str = "balanced"
    repaint_strength: float = 0.5
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = 0.0
    thinking: bool = True
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_lyrics: bool = False
    use_cot_language: bool = True
    use_constrained_decoding: bool = True
    cot_bpm: Optional[int] = None
    cot_keyscale: str = ""
    cot_timesignature: str = ""
    cot_duration: Optional[float] = None
    cot_vocal_language: str = "unknown"
    cot_caption: str = ""
    cot_lyrics: str = ""

    batch_size: int = Field(1, ge=1)
    allow_lm_batch: bool = False
    use_random_seed: bool = True
    seeds: Optional[list[int]] = None
    lm_batch_chunk_size: int = 8
    constrained_decoding_debug: bool = False
    audio_format: str = "flac"


PARAM_FIELDS = {
    "task_type",
    "instruction",
    "reference_audio",
    "src_audio",
    "audio_codes",
    "caption",
    "global_caption",
    "lyrics",
    "instrumental",
    "vocal_language",
    "bpm",
    "keyscale",
    "timesignature",
    "duration",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
    "latent_shift",
    "latent_rescale",
    "inference_steps",
    "seed",
    "guidance_scale",
    "use_adg",
    "cfg_interval_start",
    "cfg_interval_end",
    "shift",
    "infer_method",
    "timesteps",
    "repainting_start",
    "repainting_end",
    "chunk_mask_mode",
    "repaint_latent_crossfade_frames",
    "repaint_wav_crossfade_sec",
    "repaint_mode",
    "repaint_strength",
    "audio_cover_strength",
    "cover_noise_strength",
    "thinking",
    "lm_temperature",
    "lm_cfg_scale",
    "lm_top_k",
    "lm_top_p",
    "lm_negative_prompt",
    "use_cot_metas",
    "use_cot_caption",
    "use_cot_lyrics",
    "use_cot_language",
    "use_constrained_decoding",
    "cot_bpm",
    "cot_keyscale",
    "cot_timesignature",
    "cot_duration",
    "cot_vocal_language",
    "cot_caption",
    "cot_lyrics",
}

CONFIG_FIELDS = {
    "batch_size",
    "allow_lm_batch",
    "use_random_seed",
    "seeds",
    "lm_batch_chunk_size",
    "constrained_decoding_debug",
    "audio_format",
}


@dataclass
class AceStepJob:
    id: str
    status: str
    state: str
    ts: float
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    task_type: str = "text2music"
    config_path: str = "acestep-v15-turbo"
    lm_model_path: str = "acestep-5Hz-lm-1.7B"
    backend: str = "pt"
    output_dir: str = ""
    output_paths: list[str] = field(default_factory=list)
    file_urls: list[str] = field(default_factory=list)
    primary_output_path: str = ""
    primary_file_url: str = ""
    log_file: str = ""
    request_file: str = ""
    result_file: str = ""
    status_message: str = ""
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    duration: float = -1.0
    audio_format: str = "flac"


def get_ready_payload() -> dict[str, Any]:
    download_ready = READY_FLAG.exists() and all(path.exists() for path in EXPECTED_MODEL_DIRS)
    env_ready = ENV_FLAG.exists() and ACE_STEP_PYTHON.exists()
    repo_ready = ACE_STEP_ROOT.exists()
    return {
        "ready": download_ready and env_ready and repo_ready,
        "download_ready": download_ready,
        "env_ready": env_ready,
        "repo_ready": repo_ready,
        "python_path": str(ACE_STEP_PYTHON),
        "repo_path": str(ACE_STEP_ROOT),
        "checkpoint_dir": str(ACE_STEP_CHECKPOINTS),
        "ready_flag": str(READY_FLAG),
        "env_flag": str(ENV_FLAG),
    }


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _status_file(job_id: str) -> Path:
    return _job_dir(job_id) / "job_status.json"


def _request_file(job_id: str) -> Path:
    return _job_dir(job_id) / "request.json"


def _result_file(job_id: str) -> Path:
    return _job_dir(job_id) / "result.json"


def _file_url_for(path_str: str) -> str:
    path = Path(path_str)
    try:
        relative = path.resolve().relative_to(JOBS_BASE.resolve())
    except ValueError:
        return ""
    if BASE_URL:
        return f"{BASE_URL}/jobs/{relative.as_posix()}"
    return f"/jobs/{relative.as_posix()}"


class _AceStepService:
    def __init__(self):
        self.jobs_root = JOBS_ROOT.resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, AceStepJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None

    def _persist(self, job: AceStepJob) -> None:
        job_dir = _job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        _status_file(job.id).write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    def _read_persisted(self, job_id: str) -> Optional[dict]:
        status_file = _status_file(job_id)
        if not status_file.exists():
            return None
        return json.loads(status_file.read_text(encoding="utf-8"))

    async def create_job(self, req: AceStepGenerateRequest) -> AceStepJob:
        ready = get_ready_payload()
        if not ready["ready"]:
            raise HTTPException(status_code=503, detail=ready)

        job_id = req.job_id or f"ace_step_1_5_{uuid.uuid4().hex[:12]}"
        job_dir = _job_dir(job_id)
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()

        job = AceStepJob(
            id=job_id,
            status="queued",
            state="queued",
            ts=now,
            created_at=now,
            task_type=req.task_type,
            config_path=req.config_path,
            lm_model_path=req.lm_model_path,
            backend=req.backend,
            output_dir=str(output_dir),
            log_file=str(job_dir / "job.log"),
            request_file=str(_request_file(job_id)),
            result_file=str(_result_file(job_id)),
            caption=req.caption,
            lyrics=req.lyrics,
            instrumental=req.instrumental,
            duration=req.duration,
            audio_format=req.audio_format,
        )
        self.jobs[job_id] = job
        self._persist(job)

        req_data = req.model_dump()
        worker_payload = {
            "project_root": str(ACE_STEP_ROOT),
            "checkpoint_dir": str(ACE_STEP_CHECKPOINTS),
            "output_dir": str(output_dir),
            "config_path": req.config_path,
            "lm_model_path": req.lm_model_path,
            "backend": req.backend,
            "device": req.device,
            "use_flash_attention": req.use_flash_attention,
            "compile_model": req.compile_model,
            "offload_to_cpu": req.offload_to_cpu,
            "offload_dit_to_cpu": req.offload_dit_to_cpu,
            "lm_offload_to_cpu": req.lm_offload_to_cpu,
            "quantization": req.quantization,
            "prefer_source": req.prefer_source,
            "use_mlx_dit": req.use_mlx_dit,
            "params": {key: req_data[key] for key in PARAM_FIELDS},
            "config": {key: req_data[key] for key in CONFIG_FIELDS},
        }
        _request_file(job_id).write_text(
            json.dumps(worker_payload, ensure_ascii=False, indent=2),
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

                log_path = Path(job.log_file)
                request_path = Path(job.request_file)
                result_path = Path(job.result_file)
                if result_path.exists():
                    result_path.unlink()

                try:
                    with open(log_path, "w", encoding="utf-8") as log_file:
                        log_file.write("backend: ace-step-1.5\n")
                        log_file.write(f"job_id: {job.id}\n")
                        log_file.write(f"python: {ACE_STEP_PYTHON}\n")
                        log_file.write(f"config_path: {job.config_path}\n")
                        log_file.write(f"lm_model_path: {job.lm_model_path}\n")
                        log_file.write(f"task_type: {job.task_type}\n")
                        log_file.write(f"output_dir: {job.output_dir}\n\n")
                        log_file.flush()

                        env = os.environ.copy()
                        env["ACE_STEP_JOB_JSON"] = str(request_path)
                        env["ACE_STEP_RESULT_JSON"] = str(result_path)
                        env["PYTHONNOUSERSITE"] = "1"
                        existing_pythonpath = env.get("PYTHONPATH", "")
                        env["PYTHONPATH"] = f"{ACE_STEP_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(ACE_STEP_ROOT)

                        proc = await asyncio.create_subprocess_exec(
                            str(ACE_STEP_PYTHON),
                            "-c",
                            WORKER_CODE,
                            cwd=str(ACE_STEP_ROOT),
                            env=env,
                            stdout=log_file,
                            stderr=log_file,
                        )
                        rc = await proc.wait()
                        job.exit_code = rc

                    if rc != 0:
                        raise RuntimeError(f"ace-step worker exited with code {rc}")
                    if not result_path.exists():
                        raise RuntimeError("ace-step worker did not write result.json")

                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    output_paths = result.get("output_paths") or []
                    if not output_paths:
                        raise RuntimeError("ace-step result did not include output_paths")

                    job.output_paths = output_paths
                    job.file_urls = [url for url in (_file_url_for(path) for path in output_paths) if url]
                    job.primary_output_path = result.get("primary_output_path") or output_paths[0]
                    job.primary_file_url = job.file_urls[0] if job.file_urls else ""
                    job.status_message = result.get("status_message", "")
                    job.status = job.state = "succeeded"
                    job.error = None
                except Exception as exc:
                    job.status = job.state = "failed"
                    job.error = str(exc)
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        lines = [line.strip() for line in reversed(log_text.splitlines()) if line.strip()]
                        if lines:
                            job.error = lines[0]
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
                "task_type": info.get("task_type"),
                "config_path": info.get("config_path"),
                "lm_model_path": info.get("lm_model_path"),
                "backend": info.get("backend"),
                "output_paths": info.get("output_paths", []),
                "primary_output_path": info.get("primary_output_path"),
                "file_urls": info.get("file_urls", []),
                "file_url": info.get("primary_file_url"),
                "log_file": info.get("log_file"),
                "status_message": info.get("status_message"),
            }
        return {
            "ok": False,
            "job_id": job_id,
            "status": info.get("status"),
            "state": info.get("state"),
            "error": info.get("error"),
            "log_file": info.get("log_file"),
        }


_service = _AceStepService()


@router.post("/generate")
async def ace_step_generate(req: AceStepGenerateRequest):
    job = await _service.create_job(req)
    status_url = f"{BASE_URL}/Ace_step_1.5/status/{job.id}" if BASE_URL else f"/Ace_step_1.5/status/{job.id}"
    get_url = f"{BASE_URL}/Ace_step_1.5/get/{job.id}" if BASE_URL else f"/Ace_step_1.5/get/{job.id}"
    return {
        "job_id": job.id,
        "state": job.state,
        "status_url": status_url,
        "get_url": get_url,
    }


@router.get("/status/{job_id}")
def ace_step_status(job_id: str):
    return _service.get_status(job_id)


@router.get("/get/{job_id}")
def ace_step_get(job_id: str):
    return _service.get_result(job_id)
