from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve


AUDIO_KEYS = ("audio_url", "artifact_url", "file_url", "output_path", "primary_output_path", "path")


def is_riff_wav(path: str | Path) -> bool:
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            return header.startswith(b"RIFF") and header[8:12] == b"WAVE"
    except OSError:
        return False


def _looks_like_json(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            start = handle.read(64).lstrip()
        return start.startswith(b"{") or start.startswith(b"[")
    except OSError:
        return False


def _walk_audio_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in AUDIO_KEYS and isinstance(item, str) and item:
                candidates.append(item)
            candidates.extend(_walk_audio_candidates(item))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_walk_audio_candidates(item))
    return candidates


def resolve_ace_artifact(response_or_status_path: str | Path, destination_wav: str | Path | None = None) -> dict[str, Any]:
    src = Path(response_or_status_path)
    report: dict[str, Any] = {
        "source": str(src),
        "destination": str(destination_wav) if destination_wav else None,
        "source_exists": src.exists(),
        "source_is_json": False,
        "resolved": False,
        "resolved_path": None,
        "error": None,
        "candidates": [],
    }
    if not src.exists():
        report["error"] = "source does not exist"
        return report
    if is_riff_wav(src):
        resolved = src
    elif _looks_like_json(src):
        report["source_is_json"] = True
        payload = json.loads(src.read_text(encoding="utf-8"))
        candidates = _walk_audio_candidates(payload)
        report["candidates"] = candidates
        resolved = None
        for candidate in candidates:
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"}:
                tmp = src.parent / "_ace_resolved_download.wav"
                urlretrieve(candidate, tmp)
                if is_riff_wav(tmp):
                    resolved = tmp
                    break
                tmp.unlink(missing_ok=True)
            else:
                candidate_path = Path(candidate)
                if not candidate_path.is_absolute():
                    candidate_path = src.parent / candidate_path
                if candidate_path.exists() and is_riff_wav(candidate_path):
                    resolved = candidate_path
                    break
        if resolved is None:
            report["error"] = "JSON response did not contain a resolvable RIFF/WAV artifact"
            return report
    else:
        report["error"] = "source is neither RIFF/WAV nor JSON with an audio artifact"
        return report

    if destination_wav:
        destination = Path(destination_wav)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if resolved.resolve() != destination.resolve():
            shutil.copyfile(resolved, destination)
        if not is_riff_wav(destination):
            destination.unlink(missing_ok=True)
            report["error"] = "resolved artifact failed RIFF/WAV validation after copy"
            return report
        resolved = destination
    report["resolved"] = True
    report["resolved_path"] = str(resolved)
    return report
