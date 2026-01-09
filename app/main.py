import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .editor_api import EditRequest, render_edit
from .OVI import OVIJobRequest, submit_job, get_status, get_file, OVI_ROOT, OVI_CKPT_DIR
from .zimage import router as zimage_router

app = FastAPI(title="OVI API", version="1.0")

# ---- static exports (falls du das nutzt) ----
BASE_DIR = Path(__file__).resolve().parent.parent  # /workspace
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR), html=False), name="exports")

# ---- Routers ----
app.include_router(zimage_router, prefix="/zimage", tags=["zimage"])

# ---- Ready Flags ----
OVI_FLAG_FILE = "/workspace/status/ovi_ready"
ZIMAGE_FLAG_FILE = "/workspace/status/zimage_ready"


@app.get("/health")
def health():
    return {"status": "ok", "OVI_ROOT": OVI_ROOT, "OVI_CKPT_DIR": OVI_CKPT_DIR}


@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(OVI_FLAG_FILE)
    return {"ready": ready, "message": "OVI bereit." if ready else "OVI wird noch vorbereitet."}



# ---- Editor ----
@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)



