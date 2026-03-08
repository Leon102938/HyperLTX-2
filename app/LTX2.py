# /workspace/app/LTX2.py
import asyncio
import json
import os
import shlex
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

LTX_ROOT = "/workspace/LTX-2"
LTX_CKPT_DIR = f"{LTX_ROOT}/checkpoints"
LTX_JOBS_DIR = "/workspace/jobs"
LTX_PYTHON = "python3"
LTX_BACKEND = "ltx-2.3"

DEFAULT_CHECKPOINT_PATH = f"{LTX_CKPT_DIR}/ltx-2.3/ltx-2.3-22b-dev.safetensors"
DEFAULT_DISTILLED_LORA_PATH = f"{LTX_CKPT_DIR}/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_PATH = f"{LTX_CKPT_DIR}/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_GEMMA_ROOT = f"{LTX_CKPT_DIR}/gemma-3"
DEFAULT_DISTILLED_LORA_STRENGTH = 1.0
DEFAULT_IMAGE_FRAME_IDX = 0
DEFAULT_IMAGE_STRENGTH = 1.0
DEFAULT_IMAGE_CRF = 33
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"

SCALAR_FLAG_MAP = {
    "negative_prompt": "--negative-prompt",
    "seed": "--seed",
    "height": "--height",
    "width": "--width",
    "num_frames": "--num-frames",
    "frame_rate": "--frame-rate",
    "num_inference_steps": "--num-inference-steps",
    "video_cfg_guidance_scale": "--video-cfg-guidance-scale",
    "video_stg_guidance_scale": "--video-stg-guidance-scale",
    "video_rescale_scale": "--video-rescale-scale",
    "a2v_guidance_scale": "--a2v-guidance-scale",
    "video_skip_step": "--video-skip-step",
    "audio_cfg_guidance_scale": "--audio-cfg-guidance-scale",
    "audio_stg_guidance_scale": "--audio-stg-guidance-scale",
    "audio_rescale_scale": "--audio-rescale-scale",
    "v2a_guidance_scale": "--v2a-guidance-scale",
    "audio_skip_step": "--audio-skip-step",
}

LIST_FLAG_MAP = {
    "video_stg_blocks": "--video-stg-blocks",
    "audio_stg_blocks": "--audio-stg-blocks",
}

BOOL_FLAG_MAP = {
    "enhance_prompt": "--enhance-prompt",
}

class LTX2JobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None


@dataclass
class Job:
    id: str
    status: str  # queued | running | succeeded | failed
    state: str  # Alias für n8n 'state' Anzeige
    ts: float  # Zeitstempel für n8n
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    output_path: str = ""  # Für n8n Anzeige
    output_file: str = ""  # Interner Pfad
    log_file: str = ""
    prompt: str = ""
    overrides: Dict[str, Any] = None
    command: Optional[list[str]] = None
    backend: str = LTX_BACKEND


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "an"}


def _normalize_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        normalized[key.replace("-", "_") if isinstance(key, str) else key] = value

    if "cfg_guidance_scale" in normalized and "video_cfg_guidance_scale" not in normalized:
        normalized["video_cfg_guidance_scale"] = normalized["cfg_guidance_scale"]
    if "distilled_lora_strength" in normalized and "distilled_strength" not in normalized:
        normalized["distilled_strength"] = normalized["distilled_lora_strength"]
    return normalized


def _looks_like_single_lora(value: list[Any] | tuple[Any, ...]) -> bool:
    if not value:
        return False
    if isinstance(value[0], (list, tuple, dict)):
        return False
    return len(value) <= 2


def _coerce_lora_entry(entry: Any, default_strength: float) -> tuple[str, float]:
    if isinstance(entry, str):
        return str(Path(entry).expanduser().resolve()), float(default_strength)
    if isinstance(entry, (list, tuple)):
        if not entry:
            raise ValueError("LoRA entry is empty")
        path = entry[0]
        strength = entry[1] if len(entry) > 1 and entry[1] is not None else default_strength
        return str(Path(str(path)).expanduser().resolve()), float(strength)
    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("file") or entry.get("model") or entry.get("checkpoint_path")
        if not path:
            raise ValueError("LoRA dict requires 'path'")
        strength = entry.get("strength", default_strength)
        return str(Path(str(path)).expanduser().resolve()), float(strength)
    raise ValueError(f"Unsupported LoRA entry: {entry!r}")


def _parse_lora_entries(value: Any, default_strength: float) -> list[tuple[str, float]]:
    if value in (None, "", []):
        return []

    if isinstance(value, (str, dict)):
        raw_entries = [value]
    elif isinstance(value, (list, tuple)):
        raw_entries = [value] if _looks_like_single_lora(value) else list(value)
    else:
        raw_entries = [value]

    entries: list[tuple[str, float]] = []
    for entry in raw_entries:
        entries.append(_coerce_lora_entry(entry, default_strength))
    return entries


def _coerce_image_entry(
    entry: Any,
    default_frame_idx: int,
    default_strength: float,
    default_crf: int,
) -> tuple[str, int, float, Optional[int]]:
    if isinstance(entry, str):
        return str(Path(entry).expanduser().resolve()), default_frame_idx, float(default_strength), default_crf

    if isinstance(entry, (list, tuple)):
        if not entry:
            raise ValueError("Image entry is empty")
        if len(entry) == 1:
            return str(Path(str(entry[0])).expanduser().resolve()), default_frame_idx, float(default_strength), default_crf
        if len(entry) == 2:
            return str(Path(str(entry[0])).expanduser().resolve()), default_frame_idx, float(entry[1]), default_crf
        if len(entry) == 3:
            return str(Path(str(entry[0])).expanduser().resolve()), int(entry[1]), float(entry[2]), default_crf
        return str(Path(str(entry[0])).expanduser().resolve()), int(entry[1]), float(entry[2]), int(entry[3])

    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("image") or entry.get("image_path") or entry.get("img_path")
        if not path:
            raise ValueError("Image dict requires 'path'")
        frame_idx = entry.get("frame_idx", entry.get("frame", default_frame_idx))
        strength = entry.get("strength", entry.get("image_strength", entry.get("img_strength", default_strength)))
        crf = entry.get("crf", entry.get("image_crf", entry.get("img_crf", default_crf)))
        return str(Path(str(path)).expanduser().resolve()), int(frame_idx), float(strength), int(crf) if crf is not None else None

    raise ValueError(f"Unsupported image entry: {entry!r}")


def _parse_image_entries(overrides: Dict[str, Any]) -> list[tuple[str, int, float, Optional[int]]]:
    default_frame_idx = int(overrides.get("image_frame_idx", overrides.get("img_frame_idx", DEFAULT_IMAGE_FRAME_IDX)))
    default_strength = float(overrides.get("image_strength", overrides.get("img_strength", DEFAULT_IMAGE_STRENGTH)))
    default_crf = int(overrides.get("image_crf", overrides.get("img_crf", DEFAULT_IMAGE_CRF)))

    raw_entries: list[Any] = []
    if overrides.get("images"):
        images = overrides["images"]
        if isinstance(images, (list, tuple)):
            raw_entries.extend(images)
        else:
            raw_entries.append(images)

    if overrides.get("image"):
        raw_entries.append(overrides["image"])

    single_image_path = overrides.get("image_path") or overrides.get("img_path")
    if single_image_path:
        raw_entries.append(
            {
                "path": single_image_path,
                "frame_idx": default_frame_idx,
                "strength": default_strength,
                "crf": default_crf,
            }
        )

    keyframes = overrides.get("keyframes")
    if isinstance(keyframes, dict):
        for frame_idx, value in keyframes.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("frame_idx", int(frame_idx))
                raw_entries.append(item)
            else:
                raw_entries.append(
                    {
                        "path": value,
                        "frame_idx": int(frame_idx),
                        "strength": default_strength,
                        "crf": default_crf,
                    }
                )
    elif keyframes:
        if isinstance(keyframes, (list, tuple)):
            raw_entries.extend(keyframes)
        else:
            raw_entries.append(keyframes)

    images: list[tuple[str, int, float, Optional[int]]] = []
    for entry in raw_entries:
        images.append(_coerce_image_entry(entry, default_frame_idx, default_strength, default_crf))
    return images


def _append_scalar_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def _append_multi_value_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value in (None, "", []):
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        cmd.append(flag)
        cmd.extend(str(item) for item in value)
        return
    cmd.extend([flag, str(value)])


def _append_raw_flags(cmd: list[str], value: Any) -> None:
    if value in (None, "", []):
        return
    if isinstance(value, str):
        cmd.extend(shlex.split(value))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                cmd.append(item)
            elif isinstance(item, (list, tuple)):
                cmd.extend(str(part) for part in item)
            else:
                cmd.append(str(item))
        return
    raise ValueError(f"Unsupported raw_flags value: {value!r}")


def _build_command(prompt: str, output_file: str, overrides: Dict[str, Any]) -> tuple[list[str], Dict[str, str]]:
    ov = _normalize_overrides(overrides)

    checkpoint_path = str(ov.get("checkpoint_path") or DEFAULT_CHECKPOINT_PATH)
    spatial_upsampler_path = str(ov.get("spatial_upsampler_path") or DEFAULT_SPATIAL_UPSAMPLER_PATH)
    gemma_root = str(ov.get("gemma_root") or DEFAULT_GEMMA_ROOT)
    distilled_default_strength = float(ov.get("distilled_strength", DEFAULT_DISTILLED_LORA_STRENGTH))
    distilled_loras = _parse_lora_entries(ov.get("distilled_lora"), distilled_default_strength)
    if not distilled_loras:
        distilled_loras = [(str(Path(DEFAULT_DISTILLED_LORA_PATH).resolve()), distilled_default_strength)]

    loras = _parse_lora_entries(ov.get("lora"), DEFAULT_DISTILLED_LORA_STRENGTH)
    for legacy_key in ("lora1", "lora2", "lora3"):
        loras.extend(_parse_lora_entries(ov.get(legacy_key), DEFAULT_DISTILLED_LORA_STRENGTH))

    images = _parse_image_entries(ov)

    cmd = [
        LTX_PYTHON,
        "-m",
        "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path",
        checkpoint_path,
        "--spatial-upsampler-path",
        spatial_upsampler_path,
        "--gemma-root",
        gemma_root,
        "--prompt",
        prompt,
        "--output-path",
        output_file,
    ]

    for path, strength in distilled_loras:
        cmd.extend(["--distilled-lora", path, str(strength)])

    for path, strength in loras:
        cmd.extend(["--lora", path, str(strength)])

    for path, frame_idx, strength, crf in images:
        cmd.extend(["--image", path, str(frame_idx), str(strength)])
        if crf is not None:
            cmd.append(str(crf))

    quantization = ov.get("quantization")
    if quantization:
        if isinstance(quantization, dict):
            policy = quantization.get("policy")
            amax_path = quantization.get("amax_path")
            if policy:
                cmd.extend(["--quantization", str(policy)])
                if amax_path:
                    cmd.append(str(Path(str(amax_path)).expanduser().resolve()))
        elif isinstance(quantization, (list, tuple)):
            if quantization:
                cmd.append("--quantization")
                cmd.extend(str(item) for item in quantization)
        else:
            cmd.extend(["--quantization", str(quantization)])

    for key, flag in SCALAR_FLAG_MAP.items():
        _append_scalar_flag(cmd, flag, ov.get(key))

    for key, flag in LIST_FLAG_MAP.items():
        _append_multi_value_flag(cmd, flag, ov.get(key))

    for key, flag in BOOL_FLAG_MAP.items():
        if _is_truthy(ov.get(key)):
            cmd.append(flag)

    _append_raw_flags(cmd, ov.get("raw_flags"))
    _append_raw_flags(cmd, ov.get("extra_flags"))

    env = os.environ.copy()
    pythonpath_entries = [
        f"{LTX_ROOT}/packages/ltx-core/src",
        f"{LTX_ROOT}/packages/ltx-pipelines/src",
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(pythonpath_entries + ([existing_pythonpath] if existing_pythonpath else []))
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(ov.get("pytorch_cuda_alloc_conf") or DEFAULT_CUDA_ALLOC_CONF)

    return cmd, env


class _LTX2Service:
    def __init__(self):
        self.jobs_root = Path(LTX_JOBS_DIR).resolve()
        self.jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def _persist(self, job: Job):
        try:
            job_dir = self.jobs_root / job.id
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "job_status.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        except Exception:
            pass

    async def create_job(self, prompt: str, overrides: Dict[str, Any], job_id: Optional[str]) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / jid
        job_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            id=jid,
            status="queued",
            state="queued",
            ts=time.time(),
            created_at=time.time(),
            output_path=f"/workspace/jobs/{jid}/{jid}.mp4",
            output_file=str(job_dir / f"{jid}.mp4"),
            log_file=str(job_dir / "job.log"),
            prompt=prompt,
            overrides=overrides or {},
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

                try:
                    cmd, env = _build_command(job.prompt, job.output_file, job.overrides or {})
                    job.command = cmd
                    self._persist(job)

                    with open(job.log_file, "w", encoding="utf-8") as log_file:
                        log_file.write(f"backend: {LTX_BACKEND}\n")
                        log_file.write(f"command: {shlex.join(cmd)}\n\n")
                        log_file.flush()

                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            cwd=LTX_ROOT,
                            env=env,
                            stdout=log_file,
                            stderr=log_file,
                        )
                        rc = await proc.wait()
                        job.exit_code = rc
                        if rc == 0:
                            job.status = job.state = "succeeded"
                            job.error = None
                        else:
                            job.status = job.state = "failed"
                            job.error = f"ltx-2.3 process exited with code {rc}"
                except Exception as exc:
                    job.status = job.state = "failed"
                    job.error = str(exc)

                job.finished_at = job.ts = time.time()
                self._persist(job)
            self._queue.task_done()


_service = _LTX2Service()


async def submit_job(req: LTX2JobRequest):
    return await _service.create_job(req.prompt, req.overrides, req.job_id)


def get_status(job_id: str):
    if job_id in _service.jobs:
        return asdict(_service.jobs[job_id])
    status_file = Path(LTX_JOBS_DIR) / job_id / "job_status.json"
    return json.loads(status_file.read_text()) if status_file.exists() else {"error": "not found"}
