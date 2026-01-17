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


# ---------------- LTX-2 ENDPUNKTE ----------------

# 1. PUSH: Job starten
@app.post("/ltx2/submit")
async def ltx2_submit(request: LTX2JobRequest):
    jid = await submit_job(request)
    return {"job_id": jid, "status_url": f"/ltx2/status/{jid}", "get_url": f"/ltx2/get/{jid}"}

# 2. POLL: Status prüfen (Warteschleife)
@app.get("/ltx2/status/{job_id}")
def ltx2_status(job_id: str):
    return get_status(job_id)

# 3. PULL (GET): Finale Infos holen (KEIN Binary, nur JSON mit Link!)
@app.get("/ltx2/get/{job_id}")
def ltx2_get(job_id: str):
    info = get_status(job_id)
    
    # Wir bauen ein sauberes Abschluss-JSON
    if info.get("status") == "succeeded":
        return {
            "ok": True,
            "job_id": job_id,
            "status": "succeeded",
            "video_url": f"/jobs/{job_id}/{job_id}.mp4",  # <--- Der Link für n8n
            "filename": f"{job_id}.mp4",
            "output_path": info.get("output_file")
        }
    else:
        # Falls man zu früh abruft oder Fehler
        return {
            "ok": False,
            "job_id": job_id,
            "status": info.get("status"),
            "error": info.get("error")
        }


# ---- Editor bleibt wie er ist ----
@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)