from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import uuid
import json
import shlex
import subprocess

EDIT_ROOT = os.getenv("EDIT_ROOT", "/workspace")
EXPORT_DIR = os.path.join(EDIT_ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

DEFAULT_W = int(os.getenv("EDIT_W", "1080"))
DEFAULT_H = int(os.getenv("EDIT_H", "1920"))
DEFAULT_FPS = float(os.getenv("EDIT_FPS", "30"))
DEFAULT_TRANS = float(os.getenv("EDIT_TRANS_DUR", "0.12"))
DEFAULT_CRF = int(os.getenv("EDIT_CRF", "18"))
DEFAULT_PRESET = os.getenv("EDIT_PRESET", "veryfast")


class Clip(BaseModel):
    path: str


class EditRequest(BaseModel):
    clips: List[Clip]
    output_name: Optional[str] = None
    transition: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    crf: Optional[int] = None
    preset: Optional[str] = None


def _run(cmd: List[str]) -> str:
    # Debug: zeigt dir das exakte ffmpeg Kommando in Logs
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout


def _probe_duration(path: str) -> float:
    # Stream-duration ist oft stabiler als format-duration bei KI-Clips
    for args in (
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path],
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
    ):
        try:
            out = subprocess.check_output(args, text=True).strip()
            d = float(out)
            if d > 0.001:
                return d
        except Exception:
            pass
    return 0.0


def _has_audio(path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "json", path],
            text=True,
        )
        data = json.loads(out)
        return bool(data.get("streams"))
    except Exception:
        return False


def _sanitize_output_name(name: Optional[str], job_id: str) -> str:
    n = (name or "").strip()
    if n.lower() in ("", "undefined", "[undefined]", "null", "[null]", "none", "[none]"):
        n = f"edit_{job_id}"
    n = re.sub(r"[^\w\-. ]+", "_", n).strip()
    if not n.lower().endswith(".mp4"):
        n += ".mp4"
    return n


def render_edit(req: EditRequest) -> Dict[str, Any]:
    if not req.clips:
        return {"ok": False, "error": "no_clips"}

    # Settings
    W = req.width or DEFAULT_W
    H = req.height or DEFAULT_H
    FPS = float(req.fps or DEFAULT_FPS)
    TRANS = float(req.transition if req.transition is not None else DEFAULT_TRANS)
    TRANS = max(0.0, min(TRANS, 1.0))
    CRF = int(req.crf or DEFAULT_CRF)
    PRESET = req.preset or DEFAULT_PRESET

    job_id = uuid.uuid4().hex[:8]
    out_name = _sanitize_output_name(req.output_name, job_id)
    out_path = os.path.join(EXPORT_DIR, out_name)

    paths = []
    durs = []
    auds = []

    for c in req.clips:
        p = c.path
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        paths.append(p)
        d = _probe_duration(p)
        if d <= 0.001:
            d = 5.0
        durs.append(d)
        auds.append(_has_audio(p))

    # ffmpeg inputs
    cmd = ["ffmpeg", "-y"]
    for p in paths:
        cmd += ["-i", p]

    # filter_complex: normalize -> insert black segment -> concat
    fc_parts = []
    concat_inputs = []

    for i in range(len(paths)):
        # Video normalize (CFR, gleiche Größe, gleiche SAR, stabile PTS)
        fc_parts.append(
            f"[{i}:v]"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={FPS},"
            f"format=yuv420p,"
            f"setpts=PTS-STARTPTS"
            f"[v{i}]"
        )

        # Audio normalize oder Silence-Fallback passend zur Clip-Dauer
        if auds[i]:
            fc_parts.append(
                f"[{i}:a]"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"aresample=async=1:first_pts=0,"
                f"asetpts=PTS-STARTPTS"
                f"[a{i}]"
            )
        else:
            fc_parts.append(
                f"anullsrc=r=48000:cl=stereo:d={durs[i]}[a{i}]"
            )

        concat_inputs += [f"[v{i}]", f"[a{i}]"]

        # Zwischen-Segment: echtes Schwarz + Silence (damit nix bröckelt)
        if i != len(paths) - 1 and TRANS > 0:
            fc_parts.append(f"color=c=black:s={W}x{H}:r={FPS}:d={TRANS}[vb{i}]")
            fc_parts.append(f"anullsrc=r=48000:cl=stereo:d={TRANS}[ab{i}]")
            concat_inputs += [f"[vb{i}]", f"[ab{i}]"]

    n_segments = len(concat_inputs) // 2  # (v,a) Paare
    fc_parts.append(
        "".join(concat_inputs) + f"concat=n={n_segments}:v=1:a=1[vout][aout]"
    )

    filter_complex = ";".join(fc_parts)

    # Re-encode (verhindert Macroblock-Müll an Cuts)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-r", str(int(round(FPS))),
        "-g", str(int(round(FPS))),      # 1s GOP -> sehr saubere Cuts
        "-keyint_min", "1",
        "-sc_threshold", "40",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]

    _run(cmd)

    rel = os.path.relpath(out_path, EDIT_ROOT)
    return {"ok": True, "output_path": rel, "output_name": out_name, "transition": TRANS}
