from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import textwrap
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

VISUAL_PROMPT_DROP_PREFIXES = (
    "Narrative role:",
    "Story beat:",
    "Keywords:",
    "Keep:",
    "Avoid:",
    "Variation intent:",
    "Shot variation:",
    "Framing:",
    "Framing hint:",
    "Prompt delta:",
    "Style bias:",
    "Keep visual anchors:",
    "Camera cues:",
)

VISUAL_PROMPT_REWRITE_PREFIXES = (
    "Opening shot:",
    "Hook focus:",
    "Visual goal:",
    "Shot intent:",
    "Style lock:",
    "Camera:",
    "Camera style:",
    "Camera motion:",
)

VISUAL_CUE_TOKENS = {
    "alarm",
    "anchor",
    "bedroom",
    "camera",
    "cinematic",
    "clock",
    "close-up",
    "closeup",
    "composition",
    "contrast",
    "depth",
    "desk",
    "editorial",
    "field",
    "focus",
    "frame",
    "framing",
    "glass",
    "hand",
    "highlight",
    "highlights",
    "interior",
    "lens",
    "lensing",
    "light",
    "lighting",
    "material",
    "motion",
    "morning",
    "nightstand",
    "pan",
    "person",
    "phone",
    "pitcher",
    "portrait",
    "pull-back",
    "pullback",
    "push-in",
    "pushin",
    "scene",
    "serene",
    "shadow",
    "shadows",
    "shot",
    "silhouette",
    "smartphone",
    "subject",
    "sunlight",
    "surface",
    "table",
    "texture",
    "tumbler",
    "water",
    "wide",
    "window",
    "wooden",
}

TEXT_PRONE_REPLACEMENTS = (
    (re.compile(r"\bscattered papers\b", re.IGNORECASE), "blank paper sheets"),
    (re.compile(r"(?<!blank )\bpaper sheets\b", re.IGNORECASE), "blank paper sheets"),
    (re.compile(r"\bwriting on (?:a |the )?notepad\b", re.IGNORECASE), "planning over a blank notepad"),
    (
        re.compile(r"\bwriting a single word on (?:a |the )?minimalist notepad\b", re.IGNORECASE),
        "planning over a blank minimalist notepad",
    ),
    (
        re.compile(r"\bwriting a single task on (?:a |the )?minimalist blank notepad\b", re.IGNORECASE),
        "planning one priority on a blank minimalist notepad without visible writing",
    ),
    (
        re.compile(r"\bhand writing a single word on (?:a |the )?minimalist notepad\b", re.IGNORECASE),
        "hand planning over a blank minimalist notepad",
    ),
    (
        re.compile(r"\bwith the blank notepad, water, and phone arranged neatly\b", re.IGNORECASE),
        "with a closed notebook, water, and a phone arranged neatly",
    ),
    (re.compile(r"\bdocuments\b", re.IGNORECASE), "blank paper sheets"),
    (re.compile(r"(?<!blank )\bpaper\b", re.IGNORECASE), "blank paper"),
)


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


def compress_visual_prompt(prompt: str, *, max_clauses: int = 6) -> str:
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(prompt.split())) if part.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        was_rewritten = False
        if clause.startswith(VISUAL_PROMPT_DROP_PREFIXES):
            continue
        for prefix in VISUAL_PROMPT_REWRITE_PREFIXES:
            if clause.startswith(prefix):
                clause = clause[len(prefix):].strip()
                was_rewritten = True
                break
        cleaned = clause.strip().rstrip(".")
        if not cleaned:
            continue
        cleaned = _sanitize_text_prone_clause(cleaned)
        if not was_rewritten and not _looks_like_visual_clause(cleaned):
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        selected.append(cleaned)
        if len(selected) >= max_clauses:
            break

    if not selected:
        selected.append("clean cinematic subject focus")

    selected.append("clean cinematic composition")
    selected.append(
        "blank unlabeled surfaces, no visible text, no typography, no captions, no logos, no watermarks, no letters, no numbers, no signage, no labels, no interface, no screen content, no handwriting, no printed pages, no readable page content, no open documents"
    )
    return ". ".join(selected) + "."


def _sanitize_text_prone_clause(clause: str) -> str:
    cleaned = clause
    for pattern, replacement in TEXT_PRONE_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    lowered = cleaned.lower()
    if re.search(r"\bnotepad\b", cleaned, flags=re.IGNORECASE) and "blank" not in lowered:
        cleaned = re.sub(r"\bnotepad\b", "blank notepad", cleaned, flags=re.IGNORECASE)
        lowered = cleaned.lower()
    if "blank notepad" in lowered and not any(token in lowered for token in ("hand", "planning", "pen", "writing")):
        cleaned = re.sub(r"\bblank notepad\b", "closed notebook", cleaned, flags=re.IGNORECASE)
        lowered = cleaned.lower()
    if any(token in lowered for token in ("blank paper", "blank paper sheets")) and not any(
        token in lowered for token in ("hand", "planning", "pen", "writing", "notepad")
    ):
        cleaned = re.sub(r"\bblank paper sheets\b", "clean tabletop surface", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bblank paper\b", "clean tabletop surface", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _looks_like_visual_clause(clause: str) -> bool:
    lowered = clause.lower()
    token_score = sum(1 for token in VISUAL_CUE_TOKENS if token in lowered)
    comma_count = clause.count(",")
    return token_score >= 2 or (token_score >= 1 and comma_count >= 2)


def format_srt_timestamp(seconds: float) -> str:
    bounded = max(0.0, seconds)
    total_millis = int(round(bounded * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _normalize_subtitle_words(text: str) -> list[str]:
    return [word for word in text.replace("\n", " ").split() if word]


def split_subtitle_text(text: str, *, max_words: int = 7, max_chars: int = 42) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    punctuated_parts = [part.strip(" ,") for part in re.split(r"(?<=[,.;:!?])\s+", normalized) if part.strip(" ,")]
    segments: list[str] = []
    for part in punctuated_parts or [normalized]:
        words = part.split()
        current: list[str] = []
        for word in words:
            candidate_words = current + [word]
            candidate_text = " ".join(candidate_words)
            if current and (len(candidate_words) > max_words or len(candidate_text) > max_chars):
                segments.append(" ".join(current))
                current = [word]
            else:
                current = candidate_words
        if current:
            segments.append(" ".join(current))
    return segments or [normalized]


def _is_short_subtitle_segment(text: str, *, min_words: int, min_chars: int) -> bool:
    words = _normalize_subtitle_words(text)
    return len(words) < min_words or len(text.strip()) < min_chars


def _merge_short_subtitle_segments(
    segments: list[str],
    *,
    max_words: int,
    max_chars: int,
    min_words: int,
    min_chars: int,
) -> list[str]:
    merged = [segment.strip() for segment in segments if segment and segment.strip()]
    if len(merged) <= 1:
        return merged

    i = 0
    soft_word_limit = max_words + 3
    soft_char_limit = max_chars + 16
    while i < len(merged):
        segment = merged[i]
        if len(merged) == 1 or not _is_short_subtitle_segment(segment, min_words=min_words, min_chars=min_chars):
            i += 1
            continue

        if i == 0:
            candidate = f"{segment} {merged[i + 1]}".strip()
            merged[i + 1] = candidate
            merged.pop(i)
            continue

        candidate = f"{merged[i - 1]} {segment}".strip()
        candidate_words = len(_normalize_subtitle_words(candidate))
        if candidate_words <= soft_word_limit and len(candidate) <= soft_char_limit:
            merged[i - 1] = candidate
            merged.pop(i)
            i -= 1
            continue

        if i + 1 < len(merged):
            merged[i + 1] = f"{segment} {merged[i + 1]}".strip()
            merged.pop(i)
            continue

        merged[i - 1] = candidate
        merged.pop(i)
        i -= 1

    return merged


def _fit_subtitle_segment_count_to_duration(
    segments: list[str],
    *,
    scene_duration: float,
    min_segment_duration_sec: float,
) -> list[str]:
    fitted = [segment.strip() for segment in segments if segment and segment.strip()]
    while len(fitted) > 1 and (scene_duration / len(fitted)) < min_segment_duration_sec:
        merge_index = min(range(len(fitted)), key=lambda idx: len(_normalize_subtitle_words(fitted[idx])))
        if merge_index == 0:
            fitted[1] = f"{fitted[0]} {fitted[1]}".strip()
            fitted.pop(0)
        else:
            fitted[merge_index - 1] = f"{fitted[merge_index - 1]} {fitted[merge_index]}".strip()
            fitted.pop(merge_index)
    return fitted


def build_scene_subtitle_entries(
    scenes: list[Any],
    *,
    max_words: int = 7,
    max_chars: int = 42,
    min_words: int = 2,
    min_chars: int = 8,
    min_segment_duration_sec: float = 1.0,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    subtitle_index = 1
    for scene in scenes:
        narration_text = " ".join((scene.narration_text or "").split())
        if not narration_text or scene.narration_start_sec is None or scene.narration_end_sec is None:
            continue

        segments = split_subtitle_text(narration_text, max_words=max_words, max_chars=max_chars)
        if not segments:
            continue

        start_sec = float(scene.narration_start_sec)
        end_sec = float(scene.narration_end_sec)
        scene_duration = max(0.25, end_sec - start_sec)
        segments = _merge_short_subtitle_segments(
            segments,
            max_words=max_words,
            max_chars=max_chars,
            min_words=min_words,
            min_chars=min_chars,
        )
        segments = _fit_subtitle_segment_count_to_duration(
            segments,
            scene_duration=scene_duration,
            min_segment_duration_sec=min_segment_duration_sec,
        )
        weights = [max(1, len(_normalize_subtitle_words(segment))) for segment in segments]
        total_weight = float(sum(weights))
        cursor = start_sec

        minimum_slice = min(min_segment_duration_sec, scene_duration / max(len(segments), 1))
        for idx, (segment, weight) in enumerate(zip(segments, weights)):
            proportion = weight / total_weight if total_weight > 0 else 1.0 / len(segments)
            remaining_segments = len(segments) - idx - 1
            remaining_floor = remaining_segments * minimum_slice
            max_for_current = max(minimum_slice, round(end_sec - cursor - remaining_floor, 3))
            segment_duration = min(max_for_current, max(minimum_slice, round(scene_duration * proportion, 3)))
            segment_end = round(min(end_sec, cursor + segment_duration), 3)
            entries.append(
                {
                    "index": subtitle_index,
                    "start_sec": round(cursor, 3),
                    "end_sec": segment_end,
                    "text": segment,
                }
            )
            subtitle_index += 1
            cursor = segment_end

        if entries:
            entries[-1]["end_sec"] = round(end_sec, 3)

    return entries


def write_srt_subtitles(path: str | Path, entries: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    lines: list[str] = []
    for entry in entries:
        lines.append(str(entry["index"]))
        lines.append(f"{format_srt_timestamp(float(entry['start_sec']))} --> {format_srt_timestamp(float(entry['end_sec']))}")
        lines.append(str(entry["text"]).strip())
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def _ffmpeg_filter_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def format_overlay_title_text(text: str, *, max_chars_per_line: int = 18, max_lines: int = 3) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""

    wrapped = textwrap.wrap(
        normalized,
        width=max_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return normalized
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    kept = wrapped[: max_lines - 1]
    tail = " ".join(wrapped[max_lines - 1 :]).strip()
    tail_wrapped = textwrap.wrap(
        tail,
        width=max_chars_per_line + 2,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if tail_wrapped:
        kept.append(tail_wrapped[0])
    return "\n".join(kept[:max_lines])


def overlay_layout_profile(text: str) -> dict[str, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    longest_line = max((len(line) for line in lines), default=0)
    line_count = max(1, len(lines))

    font_divisor = 20
    if longest_line > 18:
        font_divisor += 2
    if longest_line > 24:
        font_divisor += 2
    if line_count > 1:
        font_divisor += 1
    if line_count > 2:
        font_divisor += 1

    return {
        "font_divisor": font_divisor,
        "box_border": 16,
        "top_margin": 44 + (line_count - 1) * 6,
        "line_spacing": 10,
    }


def assemble_final_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    duration_sec: float,
    voice_path: str | Path | None = None,
    music_path: str | Path | None = None,
    subtitle_path: str | Path | None = None,
    burn_subtitles: bool = False,
    overlay_text_file: str | Path | None = None,
    overlay_duration_sec: float = 3.5,
    voice_volume: float = 1.0,
    music_volume: float = 0.18,
    music_fade_out_sec: float = 1.5,
) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)

    command = ["ffmpeg", "-y", "-i", str(video_path)]
    voice_index: int | None = None
    music_index: int | None = None
    next_input_index = 1
    if voice_path:
        voice_index = next_input_index
        command.extend(["-i", str(voice_path)])
        next_input_index += 1
    if music_path:
        music_index = next_input_index
        command.extend(["-stream_loop", "-1", "-i", str(music_path)])

    filter_parts: list[str] = []
    video_filters: list[str] = []

    if burn_subtitles and subtitle_path:
        subtitle_style = (
            "FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H80101010,BackColour=&H50000000,BorderStyle=3,"
            "Outline=1,Shadow=0,MarginV=34,Alignment=2"
        )
        escaped_subtitle_path = _ffmpeg_filter_escape(str(Path(subtitle_path).resolve()))
        video_filters.append(f"subtitles='{escaped_subtitle_path}':force_style='{subtitle_style}'")

    if overlay_text_file:
        escaped_text_path = _ffmpeg_filter_escape(str(Path(overlay_text_file).resolve()))
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        font_clause = f":fontfile='{_ffmpeg_filter_escape(str(font_path))}'" if font_path.exists() else ""
        overlay_text = Path(overlay_text_file).read_text(encoding="utf-8").strip()
        overlay_profile = overlay_layout_profile(overlay_text)
        video_filters.append(
            "drawtext="
            f"textfile='{escaped_text_path}'{font_clause}:reload=0:fontcolor=white:"
            f"fontsize=h/{overlay_profile['font_divisor']}:line_spacing={overlay_profile['line_spacing']}:"
            f"box=1:boxcolor=black@0.45:boxborderw={overlay_profile['box_border']}:fix_bounds=1:"
            f"x=(w-text_w)/2:y={overlay_profile['top_margin']}:"
            f"enable='lt(t,{max(0.25, overlay_duration_sec):.2f})'"
        )

    video_map = "0:v:0"
    if video_filters:
        filter_parts.append(f"[0:v]{','.join(video_filters)}[vout]")
        video_map = "[vout]"

    audio_map: str | None = None
    if voice_index is not None and music_index is not None:
        fade_out_start = max(0.0, duration_sec - max(0.2, music_fade_out_sec))
        filter_parts.extend(
            [
                f"[{voice_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={voice_volume:.3f},apad=pad_dur={duration_sec:.3f},atrim=0:{duration_sec:.3f}[voicea]",
                f"[{music_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={music_volume:.3f},atrim=0:{duration_sec:.3f},afade=t=out:st={fade_out_start:.3f}:d={max(0.2, music_fade_out_sec):.3f}[musica]",
                "[voicea][musica]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[aout]",
            ]
        )
        audio_map = "[aout]"
    elif voice_index is not None:
        filter_parts.append(
            f"[{voice_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={voice_volume:.3f},apad=pad_dur={duration_sec:.3f},atrim=0:{duration_sec:.3f}[aout]"
        )
        audio_map = "[aout]"
    elif music_index is not None:
        fade_out_start = max(0.0, duration_sec - max(0.2, music_fade_out_sec))
        filter_parts.append(
            f"[{music_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={music_volume:.3f},atrim=0:{duration_sec:.3f},afade=t=out:st={fade_out_start:.3f}:d={max(0.2, music_fade_out_sec):.3f},alimiter=limit=0.95[aout]"
        )
        audio_map = "[aout]"

    if filter_parts:
        command.extend(["-filter_complex", ";".join(filter_parts)])

    command.extend(["-map", video_map])
    if audio_map:
        command.extend(["-map", audio_map])
    else:
        command.append("-an")

    if video_filters:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
    else:
        command.extend(["-c:v", "copy"])

    if audio_map:
        command.extend(["-c:a", "aac", "-b:a", "192k"])

    command.extend(
        [
            "-movflags",
            "+faststart",
            "-t",
            f"{duration_sec:.3f}",
            str(output),
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
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
