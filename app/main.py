import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .editor_api import EditRequest, render_edit
from .upscaler_api import (
    UpscaleVideoRequest,
    UpscaleSubmitRequest,
    upscale_video,
    submit_upscale_job,
    get_upscale_job,
    get_upscale_job_log,
)
from .LTX2 import LTX2JobRequest, submit_job, get_status
from .zimage import router as zimage_router


app = FastAPI(title="LTX-2 API", version="1.0")

# Exports für n8n (Link-basiert statt Binary)
BASE_DIR = Path("/workspace")
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")

# Mount für Jobs (damit Videos per Link abrufbar sind)
app.mount("/jobs", StaticFiles(directory="/workspace/jobs"), name="jobs")

# ---- Routers ----
app.include_router(zimage_router, prefix="/zimage", tags=["zimage"])

# Flags
INIT_FLAG = "/workspace/status/init_done"
ZIMAGE_FLAG_FILE = "/workspace/status/zimage_ready"


@app.get("/health")
def health():
    return {"status": "ok", "init_ready": os.path.exists(INIT_FLAG)}


@app.get("/DW/zimage_ready")
def dw_zimage_ready():
    ready = os.path.exists(ZIMAGE_FLAG_FILE)
    return {"ready": ready, "message": "Z-Image bereit." if ready else "Z-Image wird noch vorbereitet."}


@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(INIT_FLAG)
    return {"ready": ready, "message": "Modelle bereit." if ready else "Download läuft noch..."}


# ---------------- LTX-2 ENDPUNKTE ----------------

@app.post("/ltx2/submit")
async def ltx2_submit(request: LTX2JobRequest):
    jid = await submit_job(request)
    return {"job_id": jid, "status_url": f"/ltx2/status/{jid}", "get_url": f"/ltx2/get/{jid}"}


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
            "status": "succeeded",
            "video_url": f"/jobs/{job_id}/{job_id}.mp4",
            "filename": f"{job_id}.mp4",
            "output_path": info.get("output_file"),
        }

    return {
        "ok": False,
        "job_id": job_id,
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
