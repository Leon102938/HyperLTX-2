from __future__ import annotations

import json
import math
import os
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

KEYFRAME_VISUAL_RISK_POLICY_VERSION = "phaseB2_keyframe_visual_risk_v1"
TAKE_VISUAL_REVIEW_POLICY_VERSION = "phaseC_take_visual_review_v1"
FINAL_QUALITY_POLICY_VERSION = "phaseD_final_quality_verdict_v1"
VISUAL_RISK_FORBIDDEN_TERMS = (
    "readable text",
    "handwriting",
    "paper",
    "notebook",
    "document",
    "page",
    "screen",
    "ui",
    "interface",
    "logo",
    "label",
    "poster",
    "sign",
    "typography",
    "glyph",
    "letter",
    "number",
)
VISUAL_RISK_ACTION_PATTERNS = (
    r"\bwriting\b.*\b(?:paper|notebook|document|page)\b",
    r"\bhandwriting\b",
    r"\btyping\b.*\b(?:screen|ui|interface)\b",
    r"\breading\b.*\b(?:paper|notebook|document|page|screen)\b",
)
VISUAL_RISK_PROMPT_PATTERNS = (
    r"\bvisible\s+(?:screen|ui|interface|text|logo|label|poster|sign)\b",
    r"\b(?:screen|ui|interface)\s+facing\s+(?:camera|viewer)\b",
    r"\bopen\s+(?:notebook|document|page)\b",
    r"\b(?:paper|notebook|document|page)\s+(?:on|in|across)\s+(?:the\s+)?(?:desk|table|counter)\b",
    r"\bwriting\s+on\s+(?:paper|a\s+notebook|the\s+notebook|a\s+document|the\s+page)\b",
    r"\breadable\s+(?:text|label|logo|poster|sign|screen)\b",
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


def extract_review_frames(
    video_path: str | Path | None,
    output_dir: str | Path,
    *,
    count: int = 5,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    warnings: list[str] = []
    issues: list[str] = []
    frames: list[dict[str, Any]] = []
    if not video_path:
        return {"frames": frames, "warnings": ["video path missing"], "issues": issues, "frame_count": 0}

    source = Path(video_path)
    if not source.exists():
        return {"frames": frames, "warnings": [f"video file missing: {source}"], "issues": issues, "frame_count": 0}
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        return {"frames": frames, "warnings": ["ffmpeg/ffprobe not available"], "issues": issues, "frame_count": 0}

    probe = probe_video_technical_details(source)
    duration = probe.get("duration_sec")
    if not probe.get("ffprobe_ok") or duration is None or duration <= 0:
        warnings.append("could not determine video duration for visual review frames")
        return {"frames": frames, "warnings": warnings, "issues": issues, "frame_count": 0}

    max_count = max(1, int(count or 1))
    if duration < 1.0:
        sample_count = min(max_count, 1)
    elif duration < 2.5:
        sample_count = min(max_count, 3)
    else:
        sample_count = min(max_count, 5)
    ratios = [0.08] if sample_count == 1 else [0.08, 0.25, 0.5, 0.75, 0.92][:sample_count]
    epsilon = min(0.08, max(0.01, duration * 0.05))

    for index, ratio in enumerate(ratios, start=1):
        timestamp = max(0.0, min(duration - epsilon, duration * ratio))
        frame_path = output_path / f"frame_{index:03d}.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(frame_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        exists = frame_path.exists() and frame_path.stat().st_size > 0
        if result.returncode != 0 or not exists:
            warning = result.stderr.strip() or f"ffmpeg did not create review frame {index}"
            warnings.append(warning)
        frames.append(
            {
                "timestamp_sec": round(timestamp, 3),
                "path": str(frame_path),
                "exists": exists,
                "file_size_bytes": frame_path.stat().st_size if frame_path.exists() else None,
            }
        )

    return {
        "frames": frames,
        "warnings": _unique_strings(warnings),
        "issues": _unique_strings(issues),
        "frame_count": sum(1 for frame in frames if frame.get("exists")),
        "source_video": str(source),
        "duration_sec": round(float(duration), 3),
    }


def evaluate_take_visual_review(
    *,
    validation: Any | None,
    scene_world_contract: dict[str, Any] | None,
    review_frames: list[dict[str, Any]] | None = None,
    frame_warnings: list[str] | None = None,
    scene_id: str | None = None,
    take_id: str | None = None,
    prompt_text: str | None = None,
    prompt_variant_text: str | None = None,
    selected_keyframe_visual_risk: dict[str, Any] | None = None,
    enabled: Any | None = None,
    provider: str | None = None,
    model_dir: str | None = None,
    max_frames: int | str | None = None,
) -> dict[str, Any]:
    provider_name = _resolve_take_visual_review_provider(provider, enabled=enabled)
    if provider_name == "disabled":
        return _base_take_visual_review(
            status="needs_review",
            score=0.5,
            issues=[],
            warnings=["take visual review disabled"],
            provider="disabled",
            scene_world_contract=scene_world_contract,
            review_frames=review_frames or [],
            scene_id=scene_id,
            take_id=take_id,
        )

    heuristic = _evaluate_take_visual_review_heuristic(
        validation=validation,
        scene_world_contract=scene_world_contract,
        review_frames=review_frames or [],
        frame_warnings=frame_warnings or [],
        scene_id=scene_id,
        take_id=take_id,
        prompt_text=prompt_text,
        prompt_variant_text=prompt_variant_text,
        selected_keyframe_visual_risk=selected_keyframe_visual_risk,
    )
    if provider_name != "qwen3_vl":
        return heuristic

    qwen_result = _evaluate_take_visual_review_qwen3_vl(
        heuristic=heuristic,
        scene_world_contract=scene_world_contract or {},
        review_frames=review_frames or [],
        model_dir=model_dir,
        max_frames=max_frames,
    )
    return qwen_result


def evaluate_final_quality_verdict(
    *,
    final_output_path: str | Path | None,
    expected_width: int,
    expected_height: int,
    expected_frame_rate: float,
    expected_duration_sec: float,
    selected_scene_outputs: list[dict[str, Any]] | None = None,
    selected_scene_storyboards: list[dict[str, Any]] | None = None,
    assembly_metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    final_frame_enabled: Any | None = None,
    final_frame_provider: str | None = None,
    final_frame_model_dir: str | None = None,
    max_final_frames: int = 3,
    voice_metadata: dict[str, Any] | None = None,
    music_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    problem_scenes: list[dict[str, Any]] = []
    sources = [
        "final_mp4_technical_validation",
        "selected_scene_outputs",
        "take_visual_review",
        "keyframe_visual_risk_review",
        "assembly_metadata",
        "subtitle_overlay_metadata",
        "voice_music_metadata",
    ]
    assembly = dict(assembly_metadata or {})
    selected_outputs = list(selected_scene_outputs or [])
    selected_storyboards = list(selected_scene_storyboards or [])
    score = 0.9
    final_frame_extraction: dict[str, Any] = {"frames": [], "warnings": [], "issues": [], "frame_count": 0}
    final_frame_review: dict[str, Any] | None = None

    final_path = Path(final_output_path) if final_output_path else None
    if not final_path or not final_path.exists():
        issues.append("final.mp4 is missing")
        score = 0.0
        technical_validation = {
            "passed": False,
            "issues": ["final.mp4 is missing"],
            "warnings": [],
        }
    else:
        technical_validation = validate_video_take(
            final_path,
            expected_width=expected_width,
            expected_height=expected_height,
            expected_frame_rate=expected_frame_rate,
            expected_duration_sec=expected_duration_sec,
        )
        if not technical_validation.get("passed"):
            issues.extend(f"final technical issue: {issue}" for issue in technical_validation.get("issues") or [])
            warnings.extend(f"final technical warning: {warning}" for warning in technical_validation.get("warnings") or [])
            score -= 0.45

        if output_dir:
            final_frame_extraction = extract_review_frames(
                final_path,
                Path(output_dir) / "final_review_frames",
                count=max_final_frames,
            )
            warnings.extend(f"final frame extraction: {warning}" for warning in final_frame_extraction.get("warnings") or [])
            frame_provider = _resolve_take_visual_review_provider(final_frame_provider, enabled=final_frame_enabled)
            if frame_provider == "qwen3_vl":
                sources.append("qwen3_vl_final_frame_review")
            else:
                sources.append("heuristic_final_frame_review")
            final_frame_review = evaluate_take_visual_review(
                validation=technical_validation,
                scene_world_contract={
                    "visible_subject": "final assembled social video frames",
                    "environment": "assembled final video",
                    "action": "final video playback",
                    "allowed_props": ["clean visual frames"],
                    "forbidden_props": [
                        "readable text",
                        "screens",
                        "ui",
                        "paper",
                        "notebooks",
                        "documents",
                        "logos",
                        "labels",
                        "signs",
                        "posters",
                    ],
                    "text_risk_policy": "No readable text, no glyphs, no screens, no UI, no paper, no labels, no logos.",
                },
                review_frames=final_frame_extraction.get("frames", []),
                frame_warnings=final_frame_extraction.get("warnings", []),
                scene_id="final",
                take_id="final_mp4",
                prompt_text="Final assembled video frame review. Return strict JSON.",
                prompt_variant_text="Check final frames for postability and visible text risks.",
                enabled=final_frame_enabled,
                provider=frame_provider,
                model_dir=final_frame_model_dir,
                max_frames=max_final_frames,
            )
            final_status = final_frame_review.get("take_visual_review_status")
            if final_status == "rejected":
                issues.append("final frame review rejected the assembled video")
                score -= 0.35
            elif final_status == "needs_review":
                warnings.append("final frame review needs manual review")
                score -= 0.12
            if final_frame_review.get("provider") != "qwen3_vl":
                warnings.append("final frame review is heuristic/metadata-only; no real VLM image inference used")
                score -= 0.08

    selected_statuses: list[str] = []
    selected_scores: list[float] = []
    for scene in selected_outputs:
        scene_id = str(scene.get("scene_id") or "unknown_scene")
        status = str(scene.get("take_visual_review_status") or (scene.get("take_visual_review") or {}).get("take_visual_review_status") or "needs_review")
        selected_statuses.append(status)
        try:
            selected_scores.append(float(scene.get("postability_score", (scene.get("take_visual_review") or {}).get("postability_score", 0.5))))
        except (TypeError, ValueError):
            selected_scores.append(0.5)
        if status == "rejected":
            issues.append(f"{scene_id} selected take visual review rejected")
            problem_scenes.append({"scene_id": scene_id, "reason": "selected take visual review rejected"})
            score -= 0.35
        elif status == "needs_review":
            warnings.append(f"{scene_id} selected take needs visual review")
            problem_scenes.append({"scene_id": scene_id, "reason": "selected take needs visual review"})
            score -= 0.1
        review = scene.get("take_visual_review") or {}
        for issue in review.get("issues") or []:
            issues.append(f"{scene_id} take issue: {issue}")
        for warning in review.get("warnings") or []:
            warnings.append(f"{scene_id} take warning: {warning}")

    if selected_scores:
        score = (score * 0.55) + (sum(selected_scores) / len(selected_scores) * 0.45)

    for storyboard in selected_storyboards:
        selected = storyboard.get("selected_keyframe") or {}
        review = (selected.get("metadata") or {}).get("visual_risk_review") or storyboard.get("selected_visual_risk_review") or {}
        status = str(review.get("visual_risk_status") or "")
        scene_id = str(storyboard.get("scene_id") or selected.get("scene_id") or "unknown_scene")
        if status == "rejected":
            issues.append(f"{scene_id} selected keyframe visual risk rejected")
            problem_scenes.append({"scene_id": scene_id, "reason": "selected keyframe visual risk rejected"})
            score -= 0.22
        elif status == "needs_review":
            warnings.append(f"{scene_id} selected keyframe visual risk needs review")
            score -= 0.08

    if assembly.get("subtitle_burned"):
        warnings.append("burned subtitles introduce visible text into final video")
        score -= 0.08
    if assembly.get("overlay_text"):
        warnings.append("overlay text metadata is present; visible text risk requires review")
        score -= 0.08
    if assembly.get("subtitle_mode") == "sidecar" and assembly.get("subtitle_entry_count"):
        sources.append("sidecar_subtitles")
    if voice_metadata and voice_metadata.get("success") is False:
        warnings.append("voice metadata reports failure")
        score -= 0.05
    if music_metadata and music_metadata.get("success") is False:
        warnings.append("music metadata reports failure")
        score -= 0.03

    score = round(max(0.0, min(1.0, score)), 3)
    if issues and any("final.mp4 is missing" in issue or "final technical issue" in issue for issue in issues):
        status = "failed"
    elif any("selected take visual review rejected" in issue for issue in issues):
        status = "failed"
    elif issues:
        status = "failed" if score < 0.45 else "needs_review"
    elif warnings or score < 0.82:
        status = "needs_review"
    else:
        status = "passed"

    if status == "passed":
        recommended = "ready_to_publish_or_run_human_spot_check"
    elif status == "needs_review":
        recommended = "manual_visual_review_before_publish"
    else:
        recommended = "fix_or_rerender_problem_scenes"

    return {
        "final_quality_status": status,
        "final_postability_score": score,
        "main_issues": _unique_strings(issues),
        "warnings": _unique_strings(warnings),
        "problem_scenes": problem_scenes,
        "recommended_next_action": recommended,
        "quality_policy_version": FINAL_QUALITY_POLICY_VERSION,
        "quality_sources": _unique_strings(sources),
        "technical_validation": technical_validation,
        "selected_take_visual_status_counts": _status_counts(selected_statuses, ("passed", "needs_review", "rejected")),
        "selected_take_count": len(selected_outputs),
        "average_selected_take_postability_score": round(sum(selected_scores) / len(selected_scores), 3) if selected_scores else None,
        "final_frame_extraction": final_frame_extraction,
        "final_frame_review": final_frame_review,
    }


def _evaluate_take_visual_review_heuristic(
    *,
    validation: Any | None,
    scene_world_contract: dict[str, Any] | None,
    review_frames: list[dict[str, Any]],
    frame_warnings: list[str],
    scene_id: str | None,
    take_id: str | None,
    prompt_text: str | None,
    prompt_variant_text: str | None,
    selected_keyframe_visual_risk: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = scene_world_contract or {}
    checked_contract_fields = [
        "visible_subject",
        "environment",
        "action",
        "allowed_props",
        "forbidden_props",
        "text_risk_policy",
        "social_format_rules",
    ]
    issues: list[str] = []
    warnings: list[str] = list(frame_warnings)
    score = 0.86

    validation_payload = _model_or_dict(validation)
    if not validation_payload or validation_payload.get("passed") is not True:
        for issue in validation_payload.get("issues") or ["technical video validation did not pass"]:
            issues.append(f"technical validation issue: {issue}")
        score = min(score, 0.12)

    missing_fields = [
        field
        for field in ("visible_subject", "environment", "action", "allowed_props", "forbidden_props", "text_risk_policy")
        if not contract.get(field)
    ]
    if missing_fields:
        warnings.append(f"missing contract fields: {', '.join(missing_fields)}")
        score -= 0.08

    for field in ("visible_subject", "environment", "action"):
        hits = _positive_visual_risk_hits(str(contract.get(field) or ""))
        if hits:
            issues.append(f"{field} contains positive forbidden visual content: {', '.join(hits)}")
            score -= 0.25 if field == "action" else 0.18

    allowed_hits: list[str] = []
    for value in contract.get("allowed_props") or []:
        allowed_hits.extend(_positive_visual_risk_hits(str(value)))
    if allowed_hits:
        issues.append(f"allowed_props contains forbidden visual content: {', '.join(_unique_strings(allowed_hits))}")
        score -= 0.3

    action_hits = _pattern_hits(str(contract.get("action") or ""), VISUAL_RISK_ACTION_PATTERNS)
    if action_hits:
        issues.append(f"action requests text/screen/paper behavior: {', '.join(action_hits)}")
        score -= 0.35

    prompt_hits = _candidate_positive_prompt_risk_hits(prompt_variant_text or "")
    prompt_hits.extend(_candidate_positive_prompt_risk_hits(prompt_text or ""))
    if prompt_hits:
        warnings.append(f"take prompt contains risky positive content outside policy clauses: {', '.join(_unique_strings(prompt_hits))}")
        score -= 0.12

    keyframe_status = str((selected_keyframe_visual_risk or {}).get("visual_risk_status") or "")
    if keyframe_status == "rejected":
        issues.append("selected keyframe visual risk was rejected")
        score -= 0.35
    elif keyframe_status == "needs_review":
        warnings.append("selected keyframe visual risk needed review")
        score -= 0.1

    existing_frames = [frame for frame in review_frames if frame.get("exists")]
    if not existing_frames and validation_payload.get("passed") is True:
        warnings.append("no review frames could be extracted")
        score -= 0.2

    score = round(max(0.0, min(1.0, score)), 3)
    severe_issue = any(
        "positive forbidden visual content" in issue
        or "requests text/screen/paper behavior" in issue
        or "technical validation issue" in issue
        or "keyframe visual risk was rejected" in issue
        for issue in issues
    )
    if severe_issue:
        status = "rejected"
    elif warnings or score < 0.78:
        status = "needs_review"
    else:
        status = "passed"

    return _base_take_visual_review(
        status=status,
        score=score,
        issues=_unique_strings(issues),
        warnings=_unique_strings(warnings),
        provider="heuristic",
        scene_world_contract=contract,
        review_frames=review_frames,
        scene_id=scene_id,
        take_id=take_id,
        checked_contract_fields=checked_contract_fields,
    )


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


def evaluate_keyframe_visual_risk(
    *,
    scene_world_contract: dict[str, Any] | None,
    candidate_prompt_text: str | None = None,
    effective_prompt: str | None = None,
    scene_id: str | None = None,
    candidate_id: str | None = None,
    image_validation: Any | None = None,
) -> dict[str, Any]:
    contract = scene_world_contract or {}
    issues: list[str] = []
    warnings: list[str] = []
    checked_contract_fields = [
        "visible_subject",
        "environment",
        "action",
        "allowed_props",
        "forbidden_props",
        "text_risk_policy",
        "social_format_rules",
    ]
    checked_prompt_fields = ["candidate_prompt_text", "effective_prompt"]
    risk_score = 0

    missing_fields = [
        field
        for field in ("visible_subject", "environment", "action", "allowed_props", "forbidden_props", "text_risk_policy")
        if not contract.get(field)
    ]
    if missing_fields:
        warnings.append(f"missing contract fields: {', '.join(missing_fields)}")
        risk_score += 20

    for field in ("visible_subject", "environment", "action"):
        risk_hits = _positive_visual_risk_hits(str(contract.get(field) or ""))
        if risk_hits:
            issue = f"{field} contains positive forbidden visual content: {', '.join(risk_hits)}"
            issues.append(issue)
            risk_score += 45 if field == "action" else 35

    allowed_hits: list[str] = []
    for value in contract.get("allowed_props") or []:
        allowed_hits.extend(_positive_visual_risk_hits(str(value)))
    if allowed_hits:
        issues.append(f"allowed_props contains forbidden visual content: {', '.join(_unique_strings(allowed_hits))}")
        risk_score += 45

    action_text = str(contract.get("action") or "")
    action_pattern_hits = _pattern_hits(action_text, VISUAL_RISK_ACTION_PATTERNS)
    if action_pattern_hits:
        issues.append(f"action requests text/screen/paper behavior: {', '.join(action_pattern_hits)}")
        risk_score += 60

    prompt_hits = _candidate_positive_prompt_risk_hits(candidate_prompt_text or "")
    if prompt_hits:
        issues.append(f"candidate prompt requests risky positive content: {', '.join(prompt_hits)}")
        risk_score += 45

    effective_warnings = _candidate_positive_prompt_risk_hits(effective_prompt or "")
    if effective_warnings:
        warnings.append(f"effective prompt contains risky positive content outside policy clauses: {', '.join(effective_warnings)}")
        risk_score += 20

    validation_payload = _model_or_dict(image_validation)
    if validation_payload:
        if validation_payload.get("passed") is False:
            issues.append("technical image validation did not pass")
            risk_score += 60
        for issue in validation_payload.get("issues") or []:
            issues.append(f"technical image issue: {issue}")
        for warning in validation_payload.get("warnings") or []:
            warnings.append(f"technical image warning: {warning}")
    else:
        warnings.append("no image validation available; review is prompt/contract only")

    risk_score = min(100, risk_score)
    if any(issue for issue in issues if "positive forbidden visual content" in issue or "requests" in issue):
        status = "rejected"
    elif validation_payload and validation_payload.get("passed") is False:
        status = "rejected"
    elif risk_score >= 60:
        status = "rejected"
    elif risk_score >= 25 or missing_fields:
        status = "needs_review"
    else:
        status = "passed"

    return {
        "visual_risk_status": status,
        "risk_score": risk_score,
        "issues": _unique_strings(issues),
        "warnings": _unique_strings(warnings),
        "policy_version": KEYFRAME_VISUAL_RISK_POLICY_VERSION,
        "source": "contract_prompt_heuristic_plus_technical_image_check",
        "checked_contract_fields": checked_contract_fields,
        "checked_prompt_fields": checked_prompt_fields,
        "scene_id": scene_id,
        "candidate_id": candidate_id,
        "image_validation_used": bool(validation_payload),
        "social_tip_visual_guard": bool(contract.get("social_tip_visual_guard")),
    }


def _positive_visual_risk_hits(value: str) -> list[str]:
    lowered = value.lower()
    if not lowered:
        return []
    lowered = re.sub(r"\bhidden\s+(?:logos?|labels?|device faces?|displays?|screens?|interfaces?)\b", " ", lowered)
    lowered = re.sub(r"\bno\s+(?:readable\s+|visible\s+)?(?:text|paper|notebooks?|documents?|pages?|screens?|ui|interfaces?|logos?|labels?|posters?|signs?)\b", " ", lowered)
    lowered = re.sub(r"\bwithout\s+(?:readable\s+|visible\s+)?(?:text|paper|notebooks?|documents?|pages?|screens?|ui|interfaces?|logos?|labels?|posters?|signs?)\b", " ", lowered)
    return [term for term in VISUAL_RISK_FORBIDDEN_TERMS if re.search(rf"\b{re.escape(term)}s?\b", lowered)]


def _boolish_disabled(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "false", "off", "no", "none", "disabled"}


def _resolve_take_visual_review_provider(provider: str | None = None, *, enabled: Any | None = None) -> str:
    if enabled is not None and _boolish_disabled(enabled):
        return "disabled"
    if enabled is None and _boolish_disabled(os.environ.get("VISION_REVIEW_ENABLED", "1")):
        return "disabled"
    value = str(provider or os.environ.get("VISION_REVIEW_PROVIDER", "heuristic")).strip().lower()
    if value in {"", "none", "disabled", "off"}:
        return "disabled"
    if value == "qwen3_vl":
        return "qwen3_vl"
    return "heuristic"


def _base_take_visual_review(
    *,
    status: str,
    score: float,
    issues: list[str],
    warnings: list[str],
    provider: str,
    scene_world_contract: dict[str, Any] | None,
    review_frames: list[dict[str, Any]],
    scene_id: str | None,
    take_id: str | None,
    checked_contract_fields: list[str] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    checked_fields = checked_contract_fields or [
        "visible_subject",
        "environment",
        "action",
        "allowed_props",
        "forbidden_props",
        "text_risk_policy",
        "social_format_rules",
    ]
    problem_frames = [
        {
            "path": frame.get("path"),
            "timestamp_sec": frame.get("timestamp_sec"),
            "reason": "frame extraction failed",
        }
        for frame in review_frames
        if not frame.get("exists")
    ]
    contract = scene_world_contract or {}
    return {
        "take_visual_review_status": status if status in {"passed", "needs_review", "rejected"} else "needs_review",
        "postability_score": round(max(0.0, min(1.0, float(score))), 3),
        "issues": _unique_strings(issues),
        "warnings": _unique_strings(warnings),
        "problem_frames": problem_frames,
        "provider": provider,
        "policy_version": TAKE_VISUAL_REVIEW_POLICY_VERSION,
        "checked_contract_fields": checked_fields,
        "review_frames": review_frames,
        "scene_id": scene_id,
        "take_id": take_id,
        "scene_contract_summary": {
            "visible_subject": contract.get("visible_subject"),
            "environment": contract.get("environment"),
            "action": contract.get("action"),
            "allowed_props": contract.get("allowed_props"),
            "forbidden_props": contract.get("forbidden_props"),
        },
        "summary": summary or "heuristic take visual review",
    }


def _evaluate_take_visual_review_qwen3_vl(
    *,
    heuristic: dict[str, Any],
    scene_world_contract: dict[str, Any],
    review_frames: list[dict[str, Any]],
    model_dir: str | None = None,
    max_frames: int | str | None = None,
) -> dict[str, Any]:
    existing_frames = [frame for frame in review_frames if frame.get("exists")]
    if not existing_frames:
        result = dict(heuristic)
        result["provider"] = "qwen3_vl"
        if result.get("take_visual_review_status") != "rejected":
            result["take_visual_review_status"] = "needs_review"
        result["warnings"] = _unique_strings(list(result.get("warnings") or []) + ["qwen3_vl skipped because no review frames exist"])
        return result

    model_path = Path(model_dir or os.environ.get("VISION_REVIEW_MODEL_DIR", "/workspace/models/Qwen3-VL-4B-Instruct-FP8"))
    if not (model_path / "config.json").exists():
        result = dict(heuristic)
        result["provider"] = "qwen3_vl"
        if result.get("take_visual_review_status") != "rejected":
            result["take_visual_review_status"] = "needs_review"
        result["warnings"] = _unique_strings(list(result.get("warnings") or []) + [f"qwen3_vl model dir not ready: {model_path}"])
        return result

    max_frame_count = max(1, int(max_frames or os.environ.get("VISION_REVIEW_MAX_FRAMES", "3") or "3"))
    selected_frames = existing_frames[:max_frame_count]
    prompt = (
        "Review these video frames against the scene contract. Detect visible text, glyphs, screens, UI, "
        "paper, notebooks, documents, labels, logos, signs, posters, typography, letters, or numbers. "
        "Return strict JSON with status, postability_score, issues, warnings, problem_frames, summary. "
        "The status value must be exactly one of: passed, needs_review, rejected. "
        f"Scene contract: {json.dumps(scene_world_contract, ensure_ascii=True)[:2500]}"
    )
    try:
        payload = _run_qwen3_vl_review_subprocess(
            model_path=model_path,
            frames=selected_frames,
            prompt=prompt,
        )
        status = _normalize_review_status(payload.get("status") or payload.get("take_visual_review_status"))
        score = float(payload.get("postability_score", heuristic.get("postability_score", 0.5)))
        result = dict(heuristic)
        result.update(
            {
                "take_visual_review_status": status,
                "postability_score": round(max(0.0, min(1.0, score)), 3),
                "issues": _unique_strings(list(payload.get("issues") or [])),
                "warnings": _unique_strings(list(payload.get("warnings") or [])),
                "problem_frames": list(payload.get("problem_frames") or result.get("problem_frames") or []),
                "provider": "qwen3_vl",
                "real_vlm_inference_used": bool(payload.get("real_vlm_inference_used")),
                "summary": str(payload.get("summary") or "qwen3_vl take visual review"),
            }
        )
        return result
    except Exception as exc:
        result = dict(heuristic)
        result["provider"] = "qwen3_vl"
        if result.get("take_visual_review_status") != "rejected":
            result["take_visual_review_status"] = "needs_review"
        result["warnings"] = _unique_strings(list(result.get("warnings") or []) + [f"qwen3_vl inference failed: {exc}"])
        return result


def _run_qwen3_vl_review_subprocess(*, model_path: Path, frames: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    python_path = Path(os.environ.get("QWEN3_VL_PYTHON", "/workspace/venvs/qwen3-vl-review/bin/python"))
    script_path = Path(os.environ.get("QWEN3_VL_REVIEW_SCRIPT", "/workspace/scripts/qwen3_vl_review_subprocess.py"))
    timeout_sec = int(os.environ.get("QWEN3_VL_REVIEW_TIMEOUT_SEC", "240") or "240")
    if not python_path.exists():
        raise RuntimeError(f"qwen3_vl python not found: {python_path}")
    if not script_path.exists():
        raise RuntimeError(f"qwen3_vl subprocess script not found: {script_path}")

    request = {
        "model_dir": str(model_path),
        "frames": [
            {
                "path": frame.get("path"),
                "timestamp_sec": frame.get("timestamp_sec"),
            }
            for frame in frames
            if frame.get("path")
        ],
        "prompt": prompt,
        "max_new_tokens": 180,
    }
    proc = subprocess.run(
        [str(python_path), str(script_path)],
        input=json.dumps(request, ensure_ascii=True),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    stdout = proc.stdout.strip()
    if not stdout:
        stderr_tail = proc.stderr.strip().splitlines()[-3:]
        raise RuntimeError(f"qwen3_vl subprocess produced no JSON; exit={proc.returncode}; stderr={' | '.join(stderr_tail)}")
    payload = json.loads(stdout.splitlines()[-1])
    if proc.returncode != 0:
        warning = "; ".join(str(item) for item in payload.get("warnings") or [])
        raise RuntimeError(warning or f"qwen3_vl subprocess failed with exit {proc.returncode}")
    return payload


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(stripped[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        raise


def _normalize_review_status(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"passed", "pass", "ok", "clean", "safe", "approved"}:
        return "passed"
    if normalized in {"rejected", "reject", "failed", "fail", "unsafe"}:
        return "rejected"
    if normalized in {"needs_review", "review", "needs_manual_review", "manual_review", "warning"}:
        return "needs_review"
    return "needs_review"


def _candidate_positive_prompt_risk_hits(prompt: str) -> list[str]:
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(prompt.split())) if part.strip()]
    hits: list[str] = []
    for clause in clauses:
        lowered = clause.lower()
        if any(
            marker in lowered
            for marker in (
                "forbidden visuals",
                "text risk policy",
                "no readable",
                "no handwriting",
                "no paper",
                "no notebook",
                "no document",
                "no screens",
                "no screen",
                "no ui",
                "no labels",
                "no logos",
                "no posters",
                "no signs",
                "no typography",
                "no glyphs",
                "no letters",
                "no numbers",
                "without screens",
                "without screen",
                "without labels",
                "without paper",
                "without text props",
                "do not introduce",
                "avoid office",
                "avoid readable",
                "free of readable",
                "clean unlabeled",
            )
        ):
            continue
        hits.extend(_pattern_hits(lowered, VISUAL_RISK_PROMPT_PATTERNS))
    return _unique_strings(hits)


def _pattern_hits(value: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, value, flags=re.IGNORECASE)]


def _model_or_dict(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _status_counts(values: list[str], known_statuses: tuple[str, ...]) -> dict[str, int]:
    counts = {status: 0 for status in known_statuses}
    counts["unknown"] = 0
    for value in values:
        status = str(value or "unknown")
        counts[status if status in counts else "unknown"] += 1
    return counts


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
