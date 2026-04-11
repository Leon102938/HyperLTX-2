from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


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


def parse_frame_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    value = rate.strip()
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        return round(num / den, 3)
    try:
        return round(float(value), 3)
    except ValueError:
        return None


def probe_video_technical_details(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "file_exists": False,
            "ffprobe_ok": False,
            "decode_ok": False,
            "file_size_bytes": None,
        }

    details: dict[str, Any] = {
        "file_exists": True,
        "ffprobe_ok": False,
        "decode_ok": False,
        "file_size_bytes": path_obj.stat().st_size,
    }

    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration:format=duration,format_name,size",
                "-of",
                "json",
                str(path_obj),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
    except FileNotFoundError:
        details["probe_error"] = "ffprobe not available"
        return details
    except Exception as exc:
        details["probe_error"] = str(exc)
        return details

    stream = (payload.get("streams") or [{}])[0]
    format_payload = payload.get("format") or {}
    fps = parse_frame_rate(stream.get("avg_frame_rate")) or parse_frame_rate(stream.get("r_frame_rate"))

    duration_sec = None
    for raw_value in (format_payload.get("duration"), stream.get("duration")):
        if raw_value in (None, "", "N/A"):
            continue
        try:
            duration_sec = round(float(raw_value), 3)
            break
        except (TypeError, ValueError):
            continue

    details.update(
        {
            "ffprobe_ok": True,
            "width": int(stream["width"]) if stream.get("width") is not None else None,
            "height": int(stream["height"]) if stream.get("height") is not None else None,
            "fps": fps,
            "duration_sec": duration_sec,
            "codec_name": stream.get("codec_name"),
            "format_name": format_payload.get("format_name"),
        }
    )

    try:
        decode_probe = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path_obj),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        details["decode_ok"] = decode_probe.returncode == 0
        if decode_probe.returncode != 0 and decode_probe.stderr.strip():
            details["decode_error"] = decode_probe.stderr.strip()
    except FileNotFoundError:
        details["decode_error"] = "ffmpeg not available"

    return details


def validate_video_take(
    path: str | Path | None,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_rate: float,
    expected_duration_sec: float,
    minimum_size_bytes: int = 1024,
    fps_tolerance: float = 1.0,
    duration_tolerance_sec: float = 0.75,
    duration_tolerance_ratio: float = 0.35,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    probe_data: dict[str, Any] = (
        probe_video_technical_details(path) if path else {"file_exists": False, "ffprobe_ok": False, "decode_ok": False}
    )

    file_exists = bool(probe_data.get("file_exists"))
    file_size_bytes = probe_data.get("file_size_bytes")
    ffprobe_ok = bool(probe_data.get("ffprobe_ok"))
    decode_ok = bool(probe_data.get("decode_ok"))
    width = probe_data.get("width")
    height = probe_data.get("height")
    fps = probe_data.get("fps")
    duration_sec = probe_data.get("duration_sec")
    duration_delta_sec = None

    if not file_exists:
        issues.append("video file is missing")
    if file_size_bytes is not None and file_size_bytes < minimum_size_bytes:
        issues.append(f"video file is trivially small ({file_size_bytes} bytes)")
    if file_exists and not ffprobe_ok:
        issues.append("ffprobe could not read the video stream")
    if file_exists and ffprobe_ok:
        if width != expected_width or height != expected_height:
            issues.append(f"expected resolution {expected_width}x{expected_height}, got {width}x{height}")
        if fps is None:
            issues.append("frame rate could not be determined")
        elif abs(fps - expected_frame_rate) > fps_tolerance:
            issues.append(f"expected fps near {expected_frame_rate}, got {fps}")
        if duration_sec is None:
            issues.append("duration could not be determined")
        else:
            duration_delta_sec = round(duration_sec - expected_duration_sec, 3)
            allowed_delta = max(duration_tolerance_sec, expected_duration_sec * duration_tolerance_ratio)
            if abs(duration_delta_sec) > allowed_delta:
                issues.append(
                    f"expected duration near {expected_duration_sec:.3f}s, got {duration_sec:.3f}s"
                )
        if not decode_ok:
            issues.append("ffmpeg decode check reported errors")
    elif probe_data.get("probe_error"):
        warnings.append(str(probe_data["probe_error"]))

    if issues:
        validation_status = "failed" if not file_exists else "rejected"
    else:
        validation_status = "passed"

    return {
        "validation_status": validation_status,
        "passed": not issues,
        "file_exists": file_exists,
        "file_size_bytes": file_size_bytes,
        "minimum_size_bytes": minimum_size_bytes,
        "ffprobe_ok": ffprobe_ok,
        "decode_ok": decode_ok,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "duration_delta_sec": duration_delta_sec,
        "codec_name": probe_data.get("codec_name"),
        "format_name": probe_data.get("format_name"),
        "expected_width": expected_width,
        "expected_height": expected_height,
        "expected_fps": expected_frame_rate,
        "expected_duration_sec": expected_duration_sec,
        "issues": issues,
        "warnings": warnings,
    }


def probe_image_technical_details(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "file_exists": False,
            "image_open_ok": False,
            "file_size_bytes": None,
        }

    details: dict[str, Any] = {
        "file_exists": True,
        "image_open_ok": False,
        "file_size_bytes": path_obj.stat().st_size,
    }

    try:
        with Image.open(path_obj) as image:
            details.update(
                {
                    "image_open_ok": True,
                    "width": int(image.width),
                    "height": int(image.height),
                    "format_name": image.format,
                    "color_mode": image.mode,
                }
            )
    except Exception as exc:
        details["image_error"] = str(exc)

    return details


def validate_image_candidate(
    path: str | Path | None,
    *,
    expected_width: int,
    expected_height: int,
    minimum_size_bytes: int = 1024,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    probe_data: dict[str, Any] = (
        probe_image_technical_details(path) if path else {"file_exists": False, "image_open_ok": False}
    )

    file_exists = bool(probe_data.get("file_exists"))
    file_size_bytes = probe_data.get("file_size_bytes")
    image_open_ok = bool(probe_data.get("image_open_ok"))
    width = probe_data.get("width")
    height = probe_data.get("height")

    if not file_exists:
        issues.append("image file is missing")
    if file_size_bytes is not None and file_size_bytes < minimum_size_bytes:
        issues.append(f"image file is trivially small ({file_size_bytes} bytes)")
    if file_exists and not image_open_ok:
        issues.append("image file could not be opened")
    if image_open_ok and (width != expected_width or height != expected_height):
        issues.append(f"expected image resolution {expected_width}x{expected_height}, got {width}x{height}")
    if probe_data.get("image_error"):
        warnings.append(str(probe_data["image_error"]))

    if issues:
        validation_status = "failed" if not file_exists else "rejected"
    else:
        validation_status = "passed"

    return {
        "validation_status": validation_status,
        "passed": not issues,
        "file_exists": file_exists,
        "file_size_bytes": file_size_bytes,
        "minimum_size_bytes": minimum_size_bytes,
        "image_open_ok": image_open_ok,
        "width": width,
        "height": height,
        "format_name": probe_data.get("format_name"),
        "color_mode": probe_data.get("color_mode"),
        "expected_width": expected_width,
        "expected_height": expected_height,
        "issues": issues,
        "warnings": warnings,
    }


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
