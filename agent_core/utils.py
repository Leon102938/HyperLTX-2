from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAFE_DIMENSION_RE = re.compile(r"^(?P<width>\d{3,5})x(?P<height>\d{3,5})$")

RESOLUTION_PROFILES: dict[str, dict[str, tuple[int, int]]] = {
    "landscape": {
        "draft": (1024, 576),
        "standard": (1216, 704),
        "high": (1344, 768),
    },
    "portrait": {
        "draft": (576, 1024),
        "standard": (704, 1216),
        "high": (768, 1344),
    },
    "square": {
        "draft": (768, 768),
        "standard": (1024, 1024),
        "high": (1216, 1216),
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: Path | str, payload: Any) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path_obj


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slugify(value: str, default: str = "job") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return cleaned[:48] or default


def build_job_id(seed_text: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{slugify(seed_text, default='job')}-{timestamp}"


def stable_seed(value: str, *, modulus: int = 2_147_483_647) -> int:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def choose_resolution(orientation: str, resolution: str) -> tuple[int, int, str]:
    match = SAFE_DIMENSION_RE.match(resolution)
    if match:
        width = int(match.group("width"))
        height = int(match.group("height"))
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Phase-1 two-stage pipelines require custom width and height divisible by 64")
        return width, height, "custom"

    orientation_key = orientation if orientation in RESOLUTION_PROFILES else "landscape"
    resolution_key = resolution if resolution in RESOLUTION_PROFILES[orientation_key] else "standard"
    width, height = RESOLUTION_PROFILES[orientation_key][resolution_key]
    return width, height, resolution_key


def estimate_speech_duration(text: str, words_per_second: float = 2.35, floor_seconds: float = 4.0) -> float:
    normalized = " ".join(text.split())
    if not normalized:
        return floor_seconds

    words = len(normalized.split())
    estimate = words / words_per_second
    punctuation_bonus = normalized.count(",") * 0.15 + normalized.count(".") * 0.25
    return round(max(floor_seconds, estimate + punctuation_bonus + 0.75), 2)


def probe_media_duration(path: str | None) -> float | None:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None

    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path_obj),
            ],
            text=True,
        ).strip()
        duration = float(output)
    except Exception:
        return None

    if duration <= 0:
        return None
    return round(duration, 3)


def compute_num_frames(duration_sec: float, frame_rate: int) -> int:
    raw_frames = max(1, int(math.ceil(duration_sec * frame_rate)))
    return max(17, int(math.ceil((raw_frames - 1) / 8) * 8 + 1))


def frame_count_to_duration_sec(num_frames: int, frame_rate: int, *, precision: int = 3) -> float:
    scale = 10**precision
    exact_duration = num_frames / frame_rate
    return math.floor(exact_duration * scale) / scale


def quantize_duration_to_frame_contract(duration_sec: float, frame_rate: int) -> tuple[int, float]:
    num_frames = compute_num_frames(duration_sec, frame_rate)
    return num_frames, frame_count_to_duration_sec(num_frames, frame_rate)


def copy_media_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    ensure_dir(target_path.parent)
    shutil.copy2(source_path, target_path)
    return target_path


def mirror_media_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source).resolve()
    target_path = Path(target)
    ensure_dir(target_path.parent)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)
    return target_path


def mux_voice_into_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    duration_sec: float,
) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            "[1:a]apad[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-t",
            f"{duration_sec:.3f}",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def concat_video_segments(video_paths: list[str | Path], output_path: str | Path) -> Path:
    if not video_paths:
        raise ValueError("concat_video_segments requires at least one input path")

    output = Path(output_path)
    ensure_dir(output.parent)

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False, encoding="utf-8") as handle:
        concat_list_path = Path(handle.name)
        handle.write("ffconcat version 1.0\n")
        for video_path in video_paths:
            resolved = Path(video_path).resolve()
            escaped = str(resolved).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        concat_list_path.unlink(missing_ok=True)

    return output


def http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    if not raw.strip():
        return {}
    return json.loads(raw)
