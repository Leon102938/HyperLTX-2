from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def command_path(name: str) -> str | None:
    return shutil.which(name)


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,duration,channels,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "path": str(path)}
    payload = json.loads(proc.stdout or "{}")
    payload["ok"] = True
    payload["path"] = str(path)
    return payload


def has_audio_stream(path: str | Path) -> bool:
    info = ffprobe_json(path)
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def media_tooling_preflight(sample_wav: str | Path | None = None, sample_mp4: str | Path | None = None) -> dict[str, Any]:
    ffmpeg = command_path("ffmpeg")
    ffprobe = command_path("ffprobe")
    report: dict[str, Any] = {
        "ffmpeg": {"path": ffmpeg, "present": bool(ffmpeg)},
        "ffprobe": {"path": ffprobe, "present": bool(ffprobe)},
        "checks": {},
        "ready": False,
    }
    if ffmpeg:
        version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        report["ffmpeg"]["version_head"] = "\n".join(version.stdout.splitlines()[:2])
    if ffprobe:
        version = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        report["ffprobe"]["version_head"] = "\n".join(version.stdout.splitlines()[:2])
    if ffprobe and sample_wav and Path(sample_wav).exists():
        report["checks"]["wav_duration_readable"] = ffprobe_json(sample_wav)
    if ffprobe and sample_mp4 and Path(sample_mp4).exists():
        report["checks"]["mp4_duration_readable"] = ffprobe_json(sample_mp4)
    report["checks"]["mp4_audio_removal_supported"] = bool(ffmpeg and ffprobe)
    report["checks"]["voice_music_mix_supported"] = bool(ffmpeg and ffprobe)
    report["checks"]["srt_mux_or_burnin_supported"] = bool(ffmpeg and ffprobe)
    report["checks"]["concat_supported"] = bool(ffmpeg and ffprobe)
    report["ready"] = bool(ffmpeg and ffprobe)
    return report


def strip_audio_command(input_mp4: str | Path, output_mp4: str | Path) -> list[str]:
    return ["ffmpeg", "-y", "-i", str(input_mp4), "-c:v", "copy", "-an", str(output_mp4)]
