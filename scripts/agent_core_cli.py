#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_BASE_URL = os.environ.get("AGENT_CORE_BASE_URL", "http://127.0.0.1:8000")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a job to the local agent-core API and poll until it finishes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/agent_core_cli.py \\\n"
            '    --idea "A quiet sunrise over a GPU pod." \\\n'
            '    --script "Status lights pulse as the system wakes up." \\\n'
            "    --duration-sec 4 \\\n"
            "    --resolution 768x448 \\\n"
            "    --use-voice \\\n"
            "    --style cinematic \\\n"
            "    --pipeline-preference ti2vid\n"
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL. Default: %(default)s")
    parser.add_argument("--job-id", help="Optional explicit job id. A timestamped id is generated if omitted.")
    parser.add_argument("--idea", default="", help="High-level idea for the job.")
    parser.add_argument("--script", default="", help="Direct narration or script text.")
    parser.add_argument("--duration-sec", type=float, help="Target duration in seconds.")
    parser.add_argument(
        "--orientation",
        choices=("landscape", "portrait", "square"),
        default="landscape",
        help="Output orientation. Default: %(default)s",
    )
    parser.add_argument(
        "--resolution",
        default="standard",
        help="Resolution label or explicit WxH. Default: %(default)s",
    )
    parser.add_argument("--voice-id", default="Ryan", help="Voice id used when voice is enabled. Default: %(default)s")
    parser.add_argument("--language", default="German", help="Narration language metadata. Default: %(default)s")
    parser.add_argument(
        "--use-voice",
        dest="use_voice",
        action="store_true",
        default=True,
        help="Enable voice generation. Default: enabled.",
    )
    parser.add_argument(
        "--no-voice",
        dest="use_voice",
        action="store_false",
        help="Disable voice generation.",
    )
    parser.add_argument(
        "--use-storyboard",
        dest="use_storyboard",
        action="store_true",
        default=False,
        help="Enable storyboard generation before video render.",
    )
    parser.add_argument(
        "--no-storyboard",
        dest="use_storyboard",
        action="store_false",
        help="Disable storyboard generation. Default: disabled.",
    )
    parser.add_argument(
        "--use-music",
        dest="use_music",
        action="store_true",
        default=False,
        help="Enable instrumental background music generation.",
    )
    parser.add_argument(
        "--no-music",
        dest="use_music",
        action="store_false",
        help="Disable background music generation. Default: disabled.",
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=("off", "sidecar", "burn"),
        default="off",
        help="Subtitle handling mode. Default: %(default)s",
    )
    parser.add_argument("--overlay-text", default="", help="Optional opening title overlay text.")
    parser.add_argument("--music-prompt", default="", help="Optional explicit music prompt override.")
    parser.add_argument("--scene-count", type=int, help="Optional forced scene count via metadata.")
    parser.add_argument("--variations-per-scene", type=int, help="Optional variation count per scene.")
    parser.add_argument("--takes-per-scene", type=int, help="Optional take count per scene.")
    parser.add_argument("--style", default="cinematic", help="Style hint. Default: %(default)s")
    parser.add_argument(
        "--pipeline-preference",
        choices=("auto", "ti2vid", "a2vid", "fast", "balanced", "quality"),
        default="auto",
        help="Preferred render pipeline. Default: %(default)s",
    )
    parser.add_argument(
        "--vision-review-provider",
        choices=("heuristic", "qwen3_vl", "none"),
        help="Optional visual review provider override sent in job metadata.",
    )
    parser.add_argument(
        "--vision-review-enabled",
        dest="vision_review_enabled",
        action="store_true",
        default=None,
        help="Enable visual review for this submitted job via metadata.",
    )
    parser.add_argument(
        "--no-vision-review",
        dest="vision_review_enabled",
        action="store_false",
        help="Disable visual review for this submitted job via metadata.",
    )
    parser.add_argument("--vision-review-model-dir", help="Optional visual review model directory sent in job metadata.")
    parser.add_argument("--vision-review-max-frames", type=int, help="Optional max review frames sent in job metadata.")
    parser.add_argument("--poll-interval-sec", type=float, default=3.0, help="Fallback polling interval. Default: %(default)s")
    parser.add_argument("--timeout-sec", type=float, default=1800.0, help="Overall wait timeout. Default: %(default)s")
    parser.add_argument("--tail-error-log-lines", type=int, default=80, help="Lines to show from relevant backend job logs on failure. Default: %(default)s")
    parser.add_argument("--no-log-tail", action="store_true", help="Do not print backend job.log tails on failure.")
    parser.add_argument("--quiet", action="store_true", help="Print only major progress and terminal summaries.")
    parser.add_argument("--verbose", action="store_true", help="Print extra local artifact paths and backend details.")
    parser.add_argument("--inspect-run", help="Summarize an existing local run id or /workspace/agent_runs path without submitting a new job.")
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print the resolved API payload before submit.",
    )
    return parser


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=body, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        detail: Any = raw
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON response: {raw[:200]!r}") from exc


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _generate_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"manual-agent-core-{stamp}"


def format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _short_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def load_json_safe(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        p = Path(path)
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_dir_for_job(job_id: str | None) -> Path | None:
    if not job_id:
        return None
    return Path("/workspace/agent_runs") / job_id


def _resolve_inspect_run(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path("/workspace/agent_runs") / value


def _artifact_path(result: dict[str, Any], key: str) -> str | None:
    for artifact in result.get("artifacts") or []:
        if artifact.get("key") == key:
            return artifact.get("path")
    return None


def _local_run_files(job_id: str | None, result: dict[str, Any] | None = None) -> tuple[Path | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = _run_dir_for_job(job_id)
    result = result or {}
    if run_dir is None and result.get("job_id"):
        run_dir = _run_dir_for_job(str(result.get("job_id")))
    state = load_json_safe(run_dir / "state.json") if run_dir else {}
    takes = load_json_safe(run_dir / "takes.json") if run_dir else {}
    local_result = load_json_safe(run_dir / "result.json") if run_dir else {}
    if not local_result:
        local_result = result
    return run_dir, state, takes, local_result


def summarize_job_payload(payload: dict[str, Any], base_url: str) -> None:
    job = payload.get("job") or {}
    metadata = job.get("metadata") or {}
    print("JOB START")
    print(f"- job_id: {job.get('job_id')}")
    if job.get("idea"):
        print(f"- idea: {_short_text(job.get('idea'))}")
    if job.get("script"):
        print(f"- script: {_short_text(job.get('script'))}")
    print(f"- duration/resolution/orientation: {job.get('duration_sec', 'auto')}s / {job.get('resolution')} / {job.get('orientation')}")
    print(f"- flags: voice={job.get('use_voice')} storyboard={job.get('use_storyboard')} music={job.get('use_music')} subtitles={metadata.get('subtitle_mode')}")
    print(f"- takes/variations: takes={metadata.get('takes_per_scene', 'auto')} variations={metadata.get('variations_per_scene', 'auto')}")
    print(
        "- vision_review: "
        f"enabled={metadata.get('vision_review_enabled', 'default')} "
        f"provider={metadata.get('vision_review_provider', 'default')} "
        f"model_dir={_short_text(metadata.get('vision_review_model_dir', 'default'), 80)} "
        f"max_frames={metadata.get('vision_review_max_frames', 'default')}"
    )
    print(f"- api: {base_url}")


def _extract_director_summary(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source in (result, state, takes):
        if not source:
            continue
        for key in (
            "director_mode",
            "director_llm_active",
            "director_fallback_reason",
            "director_llm_provider",
            "director_llm_model",
            "director_llm_endpoint",
        ):
            if summary.get(key) is None and source.get(key) is not None:
                summary[key] = source.get(key)
    for artifact in result.get("artifacts") or []:
        if artifact.get("key") == "director_output_file":
            metadata = artifact.get("metadata") or {}
            summary.setdefault("director_mode", metadata.get("director_mode"))
            summary.setdefault("director_llm_active", metadata.get("director_llm_active"))
            summary.setdefault("director_llm_model", metadata.get("director_llm_model"))
    return {k: v for k, v in summary.items() if v is not None}


def _print_director_summary(summary: dict[str, Any]) -> None:
    if not summary:
        return
    parts = []
    if "director_mode" in summary:
        parts.append(f"mode={summary['director_mode']}")
    if "director_llm_active" in summary:
        parts.append(f"active={summary['director_llm_active']}")
    if "director_llm_provider" in summary:
        parts.append(f"provider={summary['director_llm_provider']}")
    if "director_llm_model" in summary:
        parts.append(f"model={summary['director_llm_model']}")
    if "director_fallback_reason" in summary:
        parts.append(f"fallback={summary['director_fallback_reason']}")
    if parts:
        print("Director: " + " ".join(parts))


def _step_line(name: str, step: dict[str, Any]) -> str:
    bits = [f"{name}={step.get('status', 'unknown')}"]
    if step.get("backend_name"):
        bits.append(f"backend={step.get('backend_name')}")
    if step.get("backend_job_id"):
        bits.append(f"job={step.get('backend_job_id')}")
    if step.get("duration_sec") is not None:
        bits.append(f"duration={step.get('duration_sec')}s")
    if step.get("error"):
        bits.append(f"error={_short_text(step.get('error'), 90)}")
    return " ".join(bits)


def _extract_step_summary(state: dict[str, Any]) -> list[str]:
    steps = state.get("steps") or {}
    lines = []
    for name in ("voice", "storyboard", "music", "video"):
        step = steps.get(name)
        if isinstance(step, dict):
            lines.append(_step_line(name, step))
    return lines


def _iter_scene_outputs(takes: dict[str, Any], state: dict[str, Any], result: dict[str, Any]):
    for source in (takes, state.get("steps", {}).get("video", {}).get("details", {}), result.get("metadata", {})):
        for scene in source.get("scene_outputs") or []:
            if isinstance(scene, dict):
                yield scene


def _take_lines(takes: dict[str, Any], state: dict[str, Any], result: dict[str, Any], *, failures_only: bool = False) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    seen_takes: set[tuple[Any, Any, Any]] = set()
    for scene in _iter_scene_outputs(takes, state, result):
        scene_id = scene.get("scene_id")
        selected = scene.get("selected_take_id")
        if scene_id and scene_id not in seen:
            seen.add(scene_id)
            selection = scene.get("selection") or {}
            if selection:
                lines.append(
                    f"scene={scene_id} selected={selected or 'none'} selection={selection.get('technical_selection_status', 'unknown')} reason={_short_text(selection.get('selection_reason'), 90)}"
                )
        for take in scene.get("takes") or []:
            if failures_only and take.get("status") not in {"failed", "rejected"} and take.get("review_status") not in {"failed", "rejected"}:
                continue
            review = take.get("take_visual_review") or take.get("metadata", {}).get("take_visual_review") or {}
            score = take.get("postability_score") or take.get("metadata", {}).get("postability_score")
            provider = take.get("visual_review_provider") or take.get("metadata", {}).get("visual_review_provider") or review.get("provider")
            line = (
                f"take={take.get('take_id')} scene={take.get('scene_id', scene_id)} status={take.get('status')} "
                f"review={take.get('take_visual_review_status') or take.get('review_status') or review.get('take_visual_review_status', 'unknown')}"
            )
            if provider:
                line += f" provider={provider}"
            if score is not None:
                line += f" postability={score}"
            if take.get("backend_job_id"):
                line += f" backend_job={take.get('backend_job_id')}"
            if take.get("error"):
                line += f" error={_short_text(take.get('error'), 100)}"
            take_key = (take.get("scene_id") or scene_id, take.get("take_id"), line)
            if take_key not in seen_takes:
                seen_takes.add(take_key)
                lines.append(line)
    return lines


def _extract_quality_verdict(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    verdict = metadata.get("final_quality_verdict")
    if isinstance(verdict, dict):
        return verdict
    for artifact in result.get("artifacts") or []:
        artifact_verdict = (artifact.get("metadata") or {}).get("final_quality_verdict")
        if isinstance(artifact_verdict, dict):
            return artifact_verdict
    return {}


def _print_quality_summary(result: dict[str, Any]) -> None:
    verdict = _extract_quality_verdict(result)
    if not verdict:
        return
    print("QUALITY")
    print(f"- final_quality_status: {verdict.get('final_quality_status', 'unknown')}")
    print(f"- final_postability_score: {verdict.get('final_postability_score', 'unknown')}")
    issues = verdict.get("main_issues") or []
    warnings = verdict.get("warnings") or []
    if issues:
        print(f"- main_issues: {', '.join(_short_text(x, 100) for x in issues[:5])}")
    if warnings:
        print(f"- warnings: {', '.join(_short_text(x, 100) for x in warnings[:5])}")
    if verdict.get("recommended_next_action"):
        print(f"- recommended_next_action: {verdict.get('recommended_next_action')}")
    counts = verdict.get("selected_take_visual_status_counts")
    if counts:
        print(f"- selected_take_visual_counts: {counts}")
    frame_review = verdict.get("final_frame_review") or {}
    if isinstance(frame_review, dict) and frame_review:
        provider = frame_review.get("provider", "unknown")
        warnings = [str(item).lower() for item in frame_review.get("warnings") or []]
        real_vlm = provider == "qwen3_vl" and not any(
            marker in warning
            for warning in warnings
            for marker in ("skipped", "unavailable", "failed", "model dir not ready")
        )
        print(f"- final_frame_review.provider: {provider}")
        print(f"- real_vlm_inference_used: {real_vlm}")


def tail_file(path: str | Path | None, lines: int = 80) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        data = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])
    except Exception as exc:
        return f"<could not read {p}: {exc}>"


def _candidate_backend_logs(state: dict[str, Any], takes: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for step_name, step in (state.get("steps") or {}).items():
        details = step.get("details") if isinstance(step, dict) else {}
        for scene in (details or {}).get("scene_outputs") or []:
            for take in scene.get("takes") or []:
                backend_job_id = take.get("backend_job_id") or f"{state.get('job_id')}_{take.get('take_id')}_video"
                log_file = take.get("log_file") or f"/workspace/jobs/{backend_job_id}/job.log"
                candidates.append(
                    {
                        "phase": step_name,
                        "scene_id": take.get("scene_id") or scene.get("scene_id"),
                        "take_id": take.get("take_id"),
                        "backend": take.get("backend_name") or step.get("backend_name"),
                        "backend_job_id": backend_job_id,
                        "backend_error": take.get("error") or step.get("error"),
                        "log_file": log_file,
                    }
                )
    for scene in takes.get("scene_outputs") or []:
        for take in scene.get("takes") or []:
            backend_job_id = take.get("backend_job_id") or f"{takes.get('job_id')}_{take.get('take_id')}_video"
            candidates.append(
                {
                    "phase": "video",
                    "scene_id": take.get("scene_id") or scene.get("scene_id"),
                    "take_id": take.get("take_id"),
                    "backend": take.get("backend_name") or "ltx2",
                    "backend_job_id": backend_job_id,
                    "backend_error": take.get("error"),
                    "log_file": take.get("log_file") or f"/workspace/jobs/{backend_job_id}/job.log",
                }
            )
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in candidates:
        key = (item.get("backend_job_id"), item.get("log_file"))
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def _extract_failure_details(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("steps") or {}
    failed_step_name = None
    failed_step = None
    for name, step in steps.items():
        if isinstance(step, dict) and step.get("status") == "failed":
            failed_step_name = name
            failed_step = step
            break
    logs = _candidate_backend_logs(state, takes)
    failed_logs = [item for item in logs if item.get("backend_error") or (item.get("log_file") and Path(str(item.get("log_file"))).is_file())]
    primary = failed_logs[0] if failed_logs else (logs[0] if logs else {})
    return {
        "phase": failed_step_name or result.get("final_phase") or state.get("current_phase") or "unknown",
        "agent_error": result.get("message") or (failed_step or {}).get("error") or state.get("error"),
        "backend": primary.get("backend") or (failed_step or {}).get("backend_name"),
        "backend_job_id": primary.get("backend_job_id") or (failed_step or {}).get("backend_job_id"),
        "backend_error": primary.get("backend_error") or (failed_step or {}).get("error"),
        "scene_id": primary.get("scene_id"),
        "take_id": primary.get("take_id"),
        "log_file": primary.get("log_file"),
        "all_logs": logs,
    }


def _print_step_and_take_summary(state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any], *, verbose: bool = False) -> None:
    step_lines = _extract_step_summary(state)
    if step_lines:
        print("STEPS")
        for line in step_lines:
            print(f"- {line}")
    take_lines = _take_lines(takes, state, result, failures_only=not verbose)
    if take_lines:
        print("TAKES")
        for line in take_lines[:20]:
            print(f"- {line}")


def print_success_summary(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], run_dir: Path | None, base_url: str, *, verbose: bool = False) -> None:
    print("SUCCESS SUMMARY")
    print(f"- success: {result.get('success')}")
    print(f"- final_phase: {result.get('final_phase')}")
    final_path = result.get("output_final_path") or _artifact_path(result, "final_output_mp4")
    if final_path:
        size = "unknown"
        p = Path(final_path)
        if p.is_file():
            size = f"{p.stat().st_size / (1024 * 1024):.1f} MB"
        print(f"- final.mp4: {final_path} ({size})")
    print(f"- planned/voice/video/final duration: {result.get('planned_duration_sec')} / {result.get('actual_voice_duration_sec')} / {result.get('actual_video_duration_sec')} / {result.get('actual_final_duration_sec')}")
    if run_dir:
        print(f"- result.json: {run_dir / 'result.json'}")
        print(f"- state.json: {run_dir / 'state.json'}")
        if (run_dir / "takes.json").exists():
            print(f"- takes.json: {run_dir / 'takes.json'}")
    _print_director_summary(_extract_director_summary(result, state, takes))
    if verbose:
        _print_step_and_take_summary(state, takes, result, verbose=True)
    _print_quality_summary(result)


def print_failure_summary(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], run_dir: Path | None, *, tail_lines: int = 80, show_tail: bool = True) -> None:
    details = _extract_failure_details(result, state, takes)
    print("ERROR SUMMARY")
    print(f"- phase: {details.get('phase')}")
    print(f"- scene: {details.get('scene_id') or 'unknown'}")
    print(f"- take: {details.get('take_id') or 'unknown'}")
    print(f"- backend: {details.get('backend') or 'unknown'}")
    print(f"- backend_job_id: {details.get('backend_job_id') or 'unknown'}")
    print(f"- agent_error: {_short_text(details.get('agent_error'), 200)}")
    if details.get("backend_error"):
        print(f"- backend_error: {_short_text(details.get('backend_error'), 200)}")
    if details.get("log_file"):
        print(f"- log_file: {details.get('log_file')}")
    if run_dir:
        print(f"- result.json: {run_dir / 'result.json'}")
        print(f"- state.json: {run_dir / 'state.json'}")
        if (run_dir / "takes.json").exists():
            print(f"- takes.json: {run_dir / 'takes.json'}")
    _print_step_and_take_summary(state, takes, result, verbose=False)
    _print_quality_summary(result)
    if show_tail and details.get("log_file"):
        tail = tail_file(details.get("log_file"), tail_lines)
        if tail:
            print(f"BACKEND LOG TAIL ({tail_lines} lines): {details.get('log_file')}")
            print(tail)


def _status_signature(payload: dict[str, Any], state: dict[str, Any], takes: dict[str, Any]) -> tuple[Any, ...]:
    steps = state.get("steps") or {}
    step_sig = tuple((name, (steps.get(name) or {}).get("status"), (steps.get(name) or {}).get("backend_job_id")) for name in ("voice", "storyboard", "music", "video"))
    director = _extract_director_summary(payload.get("result") or {}, state, takes)
    return (
        payload.get("status"),
        payload.get("current_phase"),
        payload.get("status_summary"),
        step_sig,
        tuple(sorted(director.items())),
    )


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.idea and not args.script:
        raise RuntimeError("At least one of --idea or --script is required.")

    job: dict[str, Any] = {
        "job_id": args.job_id or _generate_job_id(),
        "idea": args.idea,
        "script": args.script,
        "resolution": args.resolution,
        "use_voice": args.use_voice,
        "use_music": args.use_music,
        "use_storyboard": args.use_storyboard,
        "style": args.style,
        "pipeline_preference": args.pipeline_preference,
        "orientation": args.orientation,
        "metadata": {
            "language": args.language,
            "subtitle_mode": args.subtitle_mode,
        },
    }

    if args.duration_sec is not None:
        job["duration_sec"] = args.duration_sec
    if args.use_voice:
        job["voice_id"] = args.voice_id
    if args.overlay_text:
        job["metadata"]["overlay_text"] = args.overlay_text
    if args.music_prompt:
        job["metadata"]["music_prompt"] = args.music_prompt
    if args.scene_count is not None:
        job["metadata"]["scene_count"] = args.scene_count
    if args.variations_per_scene is not None:
        job["metadata"]["variations_per_scene"] = args.variations_per_scene
    if args.takes_per_scene is not None:
        job["metadata"]["takes_per_scene"] = args.takes_per_scene
    if args.vision_review_enabled is not None:
        job["metadata"]["vision_review_enabled"] = args.vision_review_enabled
    if args.vision_review_provider:
        job["metadata"]["vision_review_provider"] = args.vision_review_provider
    if args.vision_review_model_dir:
        job["metadata"]["vision_review_model_dir"] = args.vision_review_model_dir
    if args.vision_review_max_frames is not None:
        job["metadata"]["vision_review_max_frames"] = args.vision_review_max_frames

    return {"job": job}


def _absolute_url(base_url: str, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    return parse.urljoin(f"{base_url}/", maybe_relative.lstrip("/"))


def _print_submit(payload: dict[str, Any], submit_response: dict[str, Any], base_url: str) -> None:
    job_id = submit_response.get("job_id") or payload["job"]["job_id"]
    poll_url = _absolute_url(base_url, submit_response.get("poll_url"))
    print(f"Submitted job_id={job_id}")
    if poll_url:
        print(f"Poll URL: {poll_url}")
    print(f"Initial status: {submit_response.get('status')} phase={submit_response.get('current_phase')}")


def _extract_director_fields(result: dict[str, Any]) -> tuple[Any, Any]:
    director_mode = result.get("director_mode")
    director_llm_active = result.get("director_llm_active")
    if director_mode is not None or director_llm_active is not None:
        return director_mode, director_llm_active

    for artifact in result.get("artifacts") or []:
        if artifact.get("key") != "director_output_file":
            continue
        metadata = artifact.get("metadata") or {}
        return metadata.get("director_mode"), metadata.get("director_llm_active")

    return None, None


def _poll_job(
    base_url: str,
    job_id: str,
    *,
    timeout_sec: float,
    fallback_poll_sec: float,
    quiet: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    poll_url = f"{base_url}/agent-core/jobs/{job_id}"
    deadline = time.monotonic() + timeout_sec
    start = time.monotonic()
    phase_started = start
    last_phase: Any = None
    last_signature: tuple[Any, ...] | None = None
    last_heartbeat = start

    while True:
        payload = _http_json(poll_url, timeout=30.0)
        run_dir, state, takes, result = _local_run_files(job_id, payload.get("result") or {})
        phase = payload.get("current_phase")
        now = time.monotonic()
        if phase != last_phase:
            phase_started = now
            last_phase = phase
        signature = _status_signature(payload, state, takes)
        heartbeat_due = now - last_heartbeat >= 30
        if signature != last_signature or heartbeat_due:
            prefix = f"[{format_elapsed(now - start)}]"
            print(
                f"{prefix} {payload.get('status')} phase={phase} "
                f"phase_elapsed={format_elapsed(now - phase_started)} summary={payload.get('status_summary')}"
            )
            director = _extract_director_summary(result, state, takes)
            if director and not quiet:
                _print_director_summary(director)
            if not quiet:
                for line in _extract_step_summary(state):
                    print(f"- {line}")
                take_lines = _take_lines(takes, state, result, failures_only=False)
                for line in take_lines[:8 if not verbose else 20]:
                    print(f"- {line}")
            last_signature = signature
            last_heartbeat = now

        if payload.get("is_terminal"):
            return payload

        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out while waiting for job {job_id} at {poll_url}")

        retry_after = payload.get("retry_after_sec")
        sleep_for = retry_after if isinstance(retry_after, (int, float)) and retry_after > 0 else fallback_poll_sec
        time.sleep(sleep_for)


def _print_terminal(
    payload: dict[str, Any],
    base_url: str,
    *,
    tail_lines: int = 80,
    show_log_tail: bool = True,
    verbose: bool = False,
) -> None:
    result = payload.get("result") or {}
    job_id = payload.get("job_id") or result.get("job_id")
    run_dir, state, takes, local_result = _local_run_files(job_id, result)
    result = local_result or result
    refs = payload.get("refs") or {}
    public_refs = payload.get("public_refs") or {}
    print(f"Terminal status: {payload.get('status')} success={payload.get('success')}")

    if payload.get("success") or result.get("success"):
        print_success_summary(result, state, takes, run_dir, base_url, verbose=verbose)
    else:
        print_failure_summary(result, state, takes, run_dir, tail_lines=tail_lines, show_tail=show_log_tail)

    for key in ("final_mp4_path", "result_json_path", "state_json_path"):
        value = refs.get(key)
        if value:
            print(f"{key}: {value}")

    for key in ("final_mp4_url", "result_json_url", "state_json_url"):
        value = public_refs.get(key) or refs.get(key)
        absolute = _absolute_url(base_url, value)
        if absolute:
            print(f"{key}: {absolute}")

    error_payload = payload.get("error")
    if error_payload:
        print(f"error: {json.dumps(error_payload, ensure_ascii=True)}")


def _inspect_run(path: Path, *, tail_lines: int, show_log_tail: bool, verbose: bool) -> int:
    run_dir = path
    result = load_json_safe(run_dir / "result.json")
    state = load_json_safe(run_dir / "state.json")
    takes = load_json_safe(run_dir / "takes.json")
    if not result and not state:
        print(f"ERROR: no result.json/state.json found under {run_dir}", file=sys.stderr)
        return 1
    print(f"INSPECT RUN {run_dir}")
    success = bool(result.get("success")) if result else state.get("status") == "done"
    if success:
        print_success_summary(result, state, takes, run_dir, DEFAULT_BASE_URL, verbose=verbose)
        return 0
    print_failure_summary(result, state, takes, run_dir, tail_lines=tail_lines, show_tail=show_log_tail)
    return 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    base_url = _normalize_base_url(args.base_url)

    try:
        if args.inspect_run:
            return _inspect_run(
                _resolve_inspect_run(args.inspect_run),
                tail_lines=args.tail_error_log_lines,
                show_log_tail=not args.no_log_tail,
                verbose=args.verbose,
            )

        payload = _build_payload(args)
        summarize_job_payload(payload, base_url)
        if args.print_payload:
            print(json.dumps(payload, indent=2, ensure_ascii=True))

        submit_response = _http_json(f"{base_url}/agent-core/jobs", method="POST", payload=payload, timeout=30.0)
        _print_submit(payload, submit_response, base_url)

        terminal_payload = _poll_job(
            base_url,
            submit_response.get("job_id") or payload["job"]["job_id"],
            timeout_sec=args.timeout_sec,
            fallback_poll_sec=args.poll_interval_sec,
            quiet=args.quiet,
            verbose=args.verbose,
        )
        _print_terminal(
            terminal_payload,
            base_url,
            tail_lines=args.tail_error_log_lines,
            show_log_tail=not args.no_log_tail,
            verbose=args.verbose,
        )
        return 0 if terminal_payload.get("success") else 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
