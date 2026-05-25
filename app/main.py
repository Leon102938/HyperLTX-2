import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

try:
    from .agent_core_api import router as agent_core_router
except ModuleNotFoundError as exc:
    if exc.name != "agent_core":
        raise
    agent_core_router = None
from .editor_api import EditRequest, render_edit
from .upscaler_api import (
    UpscaleVideoRequest,
    UpscaleSubmitRequest,
    upscale_video,
    submit_upscale_job,
    get_upscale_job,
    get_upscale_job_log,
)
from .ace_step_1_5 import get_ready_payload as ace_step_ready_payload
from .ace_step_1_5 import router as ace_step_router
from .LTX2 import LTX2JobRequest, LTX_BACKEND, submit_job, get_status
from .qwen_tts import router as qwen_tts_router
from .hidream import router as hidream_router
from .model_status import get_model_status, get_models_status
from .segment_pipeline_router import PipelineAvailability, capability_matrix, decide_segment_pipeline


app = FastAPI(title="LTX-2.3 API", version="2.3")

BASE_DIR = Path("/workspace")
RUNTIME_DIRS = {
    "exports": BASE_DIR / "exports",
    "jobs": BASE_DIR / "jobs",
}


def _ensure_runtime_dirs() -> dict[str, Path]:
    for path in RUNTIME_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIRS


runtime_dirs = _ensure_runtime_dirs()

# Exports für n8n (Link-basiert statt Binary)
app.mount("/exports", StaticFiles(directory=str(runtime_dirs["exports"])), name="exports")

# Mount für Jobs (damit Videos per Link abrufbar sind)
app.mount("/jobs", StaticFiles(directory=str(runtime_dirs["jobs"])), name="jobs")

# ---- Routers ----
app.include_router(ace_step_router, prefix="/Ace_step_1.5", tags=["Ace_step_1.5"])
app.include_router(hidream_router, prefix="/hidream", tags=["hidream"])
app.include_router(qwen_tts_router, prefix="/qwen_tts", tags=["qwen_tts"])
if agent_core_router is not None:
    app.include_router(agent_core_router)

# Flags
INIT_FLAG = "/workspace/status/init_done"
HIDREAM_FLAG_FILE = "/workspace/status/hidream_ready"
QWEN_ENV_FLAG_FILE = "/workspace/status/qwen_tts_env_ready"
QWEN_TOKENIZER_FLAG_FILE = "/workspace/status/qwen_tts_tokenizer_ready"
QWEN_MODEL_FLAG_FILE = "/workspace/status/qwen_tts_model_ready"
QWEN_RUNTIME_FLAG_FILE = "/workspace/status/qwen_tts_runtime_ready"


@app.get("/health")
def health():
    return {"status": "ok", "init_ready": os.path.exists(INIT_FLAG), "ltx_backend": LTX_BACKEND}


@app.get("/DW/hidream_ready")
def dw_hidream_ready():
    ready = os.path.exists(HIDREAM_FLAG_FILE)
    return {"ready": ready, "model": "HiDream-O1-Dev", "steps": 28, "message": "HiDream-O1-Dev bereit." if ready else "HiDream-O1-Dev wird noch vorbereitet."}


@app.get("/DW/qwen_tts_ready")
def dw_qwen_tts_ready():
    env_ready = os.path.exists(QWEN_ENV_FLAG_FILE)
    tokenizer_ready = os.path.exists(QWEN_TOKENIZER_FLAG_FILE)
    model_ready = os.path.exists(QWEN_MODEL_FLAG_FILE)
    runtime_ready = os.path.exists(QWEN_RUNTIME_FLAG_FILE)
    ready = env_ready and tokenizer_ready and model_ready and runtime_ready
    return {
        "ready": ready,
        "env_ready": env_ready,
        "tokenizer_ready": tokenizer_ready,
        "model_ready": model_ready,
        "runtime_ready": runtime_ready,
        "message": "Qwen TTS bereit." if ready else "Qwen TTS wird noch vorbereitet.",
    }


@app.get("/DW/ace_step_1_5_ready")
def dw_ace_step_ready():
    payload = ace_step_ready_payload()
    payload["message"] = "ACE-Step 1.5 bereit." if payload["ready"] else "ACE-Step 1.5 wird noch vorbereitet."
    return payload


@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(INIT_FLAG)
    return {"ready": ready, "message": "Modelle bereit." if ready else "Download läuft noch..."}


@app.get("/DW/models/status")
def dw_models_status():
    return get_models_status()


@app.get("/DW/models/{model_id}/ready")
def dw_model_ready(model_id: str):
    status = get_model_status(model_id)
    return {
        "model_id": model_id,
        "ready": status.get("ready"),
        "enabled": status.get("enabled"),
        "required": status.get("required"),
        "disabled": status.get("disabled"),
        "failed": status.get("failed"),
        "message": status.get("message"),
    }


# ---------------- LTX-2 / LTX-2.3 ENDPUNKTE ----------------

A2VID_ALLOWED_OVERRIDES = {
    "pipeline", "mode", "audio_path", "audio", "audio_file", "audio_file_path",
    "audio_start_time", "audio_max_duration", "image_path", "img_path", "image", "images",
    "image_frame_idx", "img_frame_idx", "image_strength", "img_strength", "image_crf", "img_crf",
    "negative_prompt", "seed", "height", "width", "num_frames", "frame_rate", "num_inference_steps",
    "video_cfg_guidance_scale", "video_cfg_scale", "video_stg_guidance_scale", "video_stg_scale",
    "video_rescale_scale", "a2v_guidance_scale", "video_modality_scale", "video_skip_step",
    "video_stg_blocks", "audio_cfg_guidance_scale", "audio_cfg_scale", "audio_stg_guidance_scale",
    "audio_stg_scale", "audio_rescale_scale", "v2a_guidance_scale", "audio_modality_scale",
    "audio_skip_step", "audio_stg_blocks", "quantization", "checkpoint_path", "spatial_upsampler_path",
    "gemma_root", "distilled_lora", "distilled_lora_strength", "distilled_strength", "lora",
    "enhance_prompt", "pytorch_cuda_alloc_conf", "dry_run", "validate_only",
}

TI2VID_ALLOWED_OVERRIDES = {
    "pipeline", "mode", "image_path", "img_path", "image", "images", "image_frame_idx",
    "img_frame_idx", "image_strength", "img_strength", "image_crf", "img_crf", "negative_prompt",
    "seed", "height", "width", "num_frames", "frame_rate", "num_inference_steps",
    "video_cfg_guidance_scale", "video_stg_guidance_scale", "video_rescale_scale",
    "a2v_guidance_scale", "video_skip_step", "video_stg_blocks", "audio_cfg_guidance_scale",
    "audio_stg_guidance_scale", "audio_rescale_scale", "v2a_guidance_scale", "audio_skip_step",
    "audio_stg_blocks", "quantization", "checkpoint_path", "spatial_upsampler_path", "gemma_root",
    "distilled_lora", "distilled_lora_strength", "distilled_strength", "lora", "enhance_prompt",
    "pytorch_cuda_alloc_conf", "dry_run", "validate_only",
}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _reject_unknown_overrides(overrides: dict, allowed: set[str]) -> None:
    unknown = sorted({str(key).replace("-", "_") for key in overrides} - allowed)
    if unknown:
        raise HTTPException(status_code=422, detail={"error": "unknown_override_args", "unknown": unknown})


@app.get("/ltx2/capabilities")
def ltx2_capabilities():
    return {
        "backend": LTX_BACKEND,
        "capabilities": capability_matrix(),
        "audio_policy": {
            "master_audio_default": "qwen_tts_voice_chunks",
            "ti2vid": "strip generated LTX audio; add Qwen-TTS in assembly",
            "a2vid": "condition on audio chunk; probe output audio; keep Qwen-TTS master by default",
            "lipdub": "requires reference video; probe output audio",
            "fake_voice": "never accept an LTX fake voice in final output",
        },
    }


@app.post("/ltx2/segment/submit")
async def ltx2_segment_submit(payload: dict):
    decision = decide_segment_pipeline(payload)
    return {
        "ok": not decision.blocked,
        "dry_run": _truthy(payload.get("dry_run", True)) or _truthy(payload.get("validate_only", True)),
        "job_submitted": False,
        "decision": decision.to_dict(),
        "note": "Segment endpoint performs routing validation only; submit generation through the selected pipeline endpoint.",
    }


@app.post("/ltx2/ti2vid/submit")
async def ltx2_ti2vid_submit(request: LTX2JobRequest):
    overrides = dict(request.overrides or {})
    _reject_unknown_overrides(overrides, TI2VID_ALLOWED_OVERRIDES)
    overrides["pipeline"] = "image_to_video"
    if _truthy(overrides.get("dry_run")) or _truthy(overrides.get("validate_only")):
        return {
            "ok": True,
            "dry_run": True,
            "job_submitted": False,
            "selected_pipeline": "ti2vid_two_stages",
            "native_audio_conditioning": False,
            "audio_policy": "strip_ltx_audio_use_qwen_tts_master_in_assembly",
        }
    jid = await submit_job(LTX2JobRequest(prompt=request.prompt, overrides=overrides, job_id=request.job_id))
    return {"job_id": jid, "backend": LTX_BACKEND, "status_url": f"/ltx2/status/{jid}", "get_url": f"/ltx2/get/{jid}"}


@app.post("/ltx2/submit")
async def ltx2_submit(request: LTX2JobRequest):
    jid = await submit_job(request)
    return {
        "job_id": jid,
        "backend": LTX_BACKEND,
        "status_url": f"/ltx2/status/{jid}",
        "get_url": f"/ltx2/get/{jid}",
    }


@app.post("/ltx2/a2vid/submit")
async def ltx2_a2vid_submit(request: LTX2JobRequest):
    overrides = dict(request.overrides or {})
    _reject_unknown_overrides(overrides, A2VID_ALLOWED_OVERRIDES)
    if not any(overrides.get(key) for key in ("audio_path", "audio", "audio_file", "audio_file_path")):
        raise HTTPException(status_code=422, detail={"error": "audio_path_required_for_a2vid"})
    overrides["pipeline"] = "a2vid_two_stage"
    if _truthy(overrides.get("dry_run")) or _truthy(overrides.get("validate_only")):
        return {
            "ok": True,
            "dry_run": True,
            "job_submitted": False,
            "selected_pipeline": "a2vid_two_stage",
            "native_audio_image_to_video": True,
            "strict_lipsync_guaranteed": False,
            "audio_policy": "condition_on_audio_chunk_probe_output_keep_qwen_tts_master_by_default",
        }
    jid = await submit_job(LTX2JobRequest(prompt=request.prompt, overrides=overrides, job_id=request.job_id))
    return {
        "job_id": jid,
        "backend": LTX_BACKEND,
        "native_audio_image_to_video": True,
        "status_url": f"/ltx2/status/{jid}",
        "get_url": f"/ltx2/get/{jid}",
    }


@app.post("/ltx2/lipdub/submit")
async def ltx2_lipdub_submit(payload: dict):
    availability = PipelineAvailability.detect()
    reference_video = payload.get("reference_video_path") or payload.get("video_path")
    audio_path = payload.get("audio_path")
    if not availability.lipdub:
        raise HTTPException(status_code=503, detail={"error": "lipdub_unavailable", "missing": ["ltx_pipelines.lipdub module"]})
    if not reference_video or not audio_path:
        raise HTTPException(status_code=422, detail={"error": "reference_video_and_audio_path_required"})
    return {"ok": True, "dry_run": True, "job_submitted": False, "selected_pipeline": "lipdub"}


@app.post("/ltx2/retake/submit")
async def ltx2_retake_submit(payload: dict):
    availability = PipelineAvailability.detect()
    if not availability.retake:
        raise HTTPException(status_code=503, detail={"error": "retake_unavailable"})
    if _truthy(payload.get("dry_run", True)) or _truthy(payload.get("validate_only", True)):
        return {"ok": True, "dry_run": True, "job_submitted": False, "selected_pipeline": "retake"}
    raise HTTPException(status_code=501, detail={"error": "retake_runner_not_wired_for_live_submit"})


@app.post("/ltx2/keyframe/submit")
async def ltx2_keyframe_submit(payload: dict):
    availability = PipelineAvailability.detect()
    if not availability.keyframe_interpolation:
        raise HTTPException(status_code=503, detail={"error": "keyframe_interpolation_unavailable"})
    if _truthy(payload.get("dry_run", True)) or _truthy(payload.get("validate_only", True)):
        return {"ok": True, "dry_run": True, "job_submitted": False, "selected_pipeline": "keyframe_interpolation"}
    raise HTTPException(status_code=501, detail={"error": "keyframe_runner_not_wired_for_live_submit"})


@app.get("/ltx2/status/{job_id}")
def ltx2_status(job_id: str):
    return get_status(job_id)


@app.get("/ltx2/get/{job_id}")
def ltx2_get(job_id: str):
    info = get_status(job_id)

    if info.get("status") == "succeeded":
        return {
            "ok": True,
            "job_id": job_id,
            "backend": info.get("backend", LTX_BACKEND),
            "status": "succeeded",
            "video_url": f"/jobs/{job_id}/{job_id}.mp4",
            "filename": f"{job_id}.mp4",
            "output_path": info.get("output_file"),
        }

    return {
        "ok": False,
        "job_id": job_id,
        "backend": info.get("backend", LTX_BACKEND),
        "status": info.get("status"),
        "error": info.get("error"),
    }


# ---- Editor ----
@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)


# ---- Upscale ----
@app.post("/upscale/video")
def upscale_video_route(request: UpscaleVideoRequest):
    return upscale_video(request)


@app.post("/upscale/submit")
def upscale_submit_route(request: UpscaleSubmitRequest):
    return submit_upscale_job(request)


@app.get("/upscale/get/{job_id}")
def upscale_get_route(job_id: str):
    return get_upscale_job(job_id)


@app.get("/upscale/log/{job_id}")
def upscale_log_route(job_id: str, tail: int = 120):
    return get_upscale_job_log(job_id, tail=tail)
