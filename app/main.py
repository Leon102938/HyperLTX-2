import os
from pathlib import Path

from fastapi import FastAPI
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


app = FastAPI(title="LTX-2.3 API", version="2.3")

BASE_DIR = Path("/workspace")
RUNTIME_DIRS = {
    "exports": BASE_DIR / "exports",
    "jobs": BASE_DIR / "jobs",
    "agent_runs": BASE_DIR / "agent_runs",
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
app.mount("/agent-runs", StaticFiles(directory=str(runtime_dirs["agent_runs"])), name="agent-runs")

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


# ---------------- LTX-2 / LTX-2.3 ENDPUNKTE ----------------

@app.post("/ltx2/submit")
async def ltx2_submit(request: LTX2JobRequest):
    jid = await submit_job(request)
    return {
        "job_id": jid,
        "backend": LTX_BACKEND,
        "status_url": f"/ltx2/status/{jid}",
        "get_url": f"/ltx2/get/{jid}",
    }


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
