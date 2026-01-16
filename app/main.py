# /workspace/app/main.py
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .editor_api import EditRequest, render_edit
from .LTX2 import LTX2JobRequest, submit_job, get_status # ✅ Neu LTX2

app = FastAPI(title="LTX-2 API", version="1.0")

# Exports für n8n (Link-basiert statt Binary)
BASE_DIR = Path("/workspace")
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
# Mount für Jobs (damit Videos per Link abrufbar sind)
app.mount("/jobs", StaticFiles(directory="/workspace/jobs"), name="jobs")

# Das Flag aus deiner init.sh
INIT_FLAG = "/workspace/status/init_done"

@app.get("/health")
def health():
    return {"status": "ok", "init_ready": os.path.exists(INIT_FLAG)}

@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(INIT_FLAG)
    return {"ready": ready, "message": "Modelle bereit." if ready else "Download läuft noch..."}

# ---- LTX-2 Endpoints ----
@app.post("/ltx2/submit")
async def ltx2_submit(request: LTX2JobRequest):
    jid = await submit_job(request)
    return {"job_id": jid, "status_url": f"/ltx2/status/{jid}"}

@app.get("/ltx2/status/{job_id}")
def ltx2_status(job_id: str):
    status = get_status(job_id)
    if "output_file" in status and status["status"] == "succeeded":
        # Erstellt den Link für n8n
        status["video_url"] = f"/jobs/{job_id}/{job_id}.mp4"
    return status

# ---- Editor bleibt wie er ist ----
@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)