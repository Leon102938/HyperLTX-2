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

from agent_core.resume_contract import inspect_resume_contract


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
        "--pipeline-dry-run",
        action="store_true",
        help="Set job.metadata.pipeline_dry_run=true so agent-core stops after planning, prompt audit, and gates.",
    )
    parser.add_argument(
        "--approval-gates-enabled",
        action="store_true",
        help="Set job.metadata.approval_gates_enabled=true so local approval files can block configured gates.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("scene_plan", "model_prompts", "storyboard"),
        help="Set job.metadata.stop_after for a controlled run that stops before later backend execution.",
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
    parser.add_argument("--live", action="store_true", help="Force the TTY live dashboard redraw mode.")
    parser.add_argument("--no-live", action="store_true", help="Disable the TTY live dashboard and append progress updates instead.")
    parser.add_argument("--inspect-run", help="Summarize an existing local run id or /workspace/agent_runs path without submitting a new job.")
    parser.add_argument("--inspect-checkpoints", help="Show checkpoints for an existing local run id or /workspace/agent_runs path.")
    parser.add_argument(
        "--approve-checkpoint",
        nargs=2,
        metavar=("JOB_ID_OR_PATH", "CHECKPOINT_ID"),
        help="Write a local approval file for a checkpoint.",
    )
    parser.add_argument(
        "--reject-checkpoint",
        nargs=2,
        metavar=("JOB_ID_OR_PATH", "CHECKPOINT_ID"),
        help="Write a local rejection file for a checkpoint.",
    )
    parser.add_argument("--approved-by", default="human", help="Approver name for --approve-checkpoint. Default: %(default)s")
    parser.add_argument("--rejected-by", default="human", help="Reviewer name for --reject-checkpoint. Default: %(default)s")
    parser.add_argument("--approval-note", default="", help="Optional note written into approval/rejection files.")
    parser.add_argument("--force-approval", action="store_true", help="Overwrite an existing approval/rejection file.")
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


def short_prompt(text: Any, max_len: int = 90) -> str:
    return _short_text(text, max_len) or "-"


def format_status_icon(status: Any) -> str:
    value = str(status or "").lower()
    if value in {"success", "succeeded", "done", "completed", "passed", "assembled", "created"}:
        return "[✓]"
    if value in {"running", "in_progress", "processing", "submitted"}:
        return "[>]"
    if value in {"failed", "error", "rejected"}:
        return "[x]"
    if value in {"needs_review", "warning", "warn"}:
        return "[!]"
    if value in {"waiting", "pending", "planned", "queued", "skipped"}:
        return "[-]"
    return "[-]"


def _plain_icon(status: Any) -> str:
    value = str(status or "").lower()
    if value in {"success", "succeeded", "done", "completed", "passed", "assembled", "created", "enabled", "true"}:
        return "✓"
    if value in {"failed", "error", "rejected", "false", "disabled"}:
        return "x"
    if value in {"needs_review", "warning", "warn"}:
        return "!"
    return "-"


def _line(label: str, value: Any, width: int = 13) -> str:
    text = "-" if value is None or value == "" else str(value)
    return f"  {label:<{width}} {text}"


def print_box_header(title: str, rows: list[tuple[str, Any]] | None = None) -> None:
    width = 60
    print("╭" + "─" * width + "╮")
    print("│ " + str(title).ljust(width - 1) + "│")
    if rows:
        print("├" + "─" * width + "┤")
        for label, value in rows:
            body = f"{label:<10} {str(value or '-')}"
            print("│ " + body[: width - 1].ljust(width - 1) + "│")
    print("╰" + "─" * width + "╯")


def format_file_size(path: str | Path | None) -> str:
    if not path:
        return "unknown"
    try:
        p = Path(path)
        if not p.is_file():
            return "unknown"
        size = p.stat().st_size
    except Exception:
        return "unknown"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


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


def _should_use_live(args: argparse.Namespace) -> bool:
    if getattr(args, "no_live", False):
        return False
    if getattr(args, "live", False):
        return True
    return bool(sys.stdout.isatty())


def _run_dir_for_job(job_id: str | None) -> Path | None:
    if not job_id:
        return None
    return Path("/workspace/agent_runs") / job_id


def _resolve_inspect_run(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path("/workspace/agent_runs") / value


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_checkpoint_payload(run_dir: Path) -> tuple[dict[str, Any], str]:
    checkpoint_payload = load_json_safe(run_dir / "checkpoints.json")
    if isinstance(checkpoint_payload.get("checkpoints"), dict):
        return checkpoint_payload, "checkpoints.json"
    state = load_json_safe(run_dir / "state.json")
    checkpoints = state.get("checkpoints")
    if isinstance(checkpoints, dict):
        return {
            "job_id": state.get("job_id"),
            "pipeline_id": state.get("pipeline_id"),
            "current_checkpoint_id": state.get("current_checkpoint_id"),
            "blocked_by_checkpoint_id": state.get("blocked_by_checkpoint_id"),
            "checkpoints": checkpoints,
        }, "state.json"
    return {}, "none"


def _checkpoint_approval_path(run_dir: Path, checkpoint_id: str, checkpoint: dict[str, Any]) -> Path:
    metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
    raw_path = metadata.get("approval_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.is_absolute() and _path_is_relative_to(path, run_dir):
            return path
        if not path.is_absolute():
            candidate = run_dir / path
            if _path_is_relative_to(candidate, run_dir):
                return candidate
    return run_dir / "approvals" / f"{checkpoint_id}.json"


def _compact_list(values: Any, *, limit: int = 3, text_limit: int = 96) -> str:
    if not values:
        return "-"
    if not isinstance(values, list):
        return short_prompt(values, text_limit)
    rendered = [short_prompt(item, text_limit) for item in values[:limit]]
    if len(values) > limit:
        rendered.append(f"+{len(values) - limit} more")
    return "; ".join(rendered) if rendered else "-"


def _checkpoint_artifact_text(artifacts: Any) -> str:
    if not isinstance(artifacts, list) or not artifacts:
        return "-"
    parts: list[str] = []
    for artifact in artifacts[:3]:
        if not isinstance(artifact, dict):
            continue
        key = artifact.get("key") or artifact.get("kind") or "artifact"
        path = artifact.get("path")
        parts.append(f"{key}={path}" if path else str(key))
    if len(artifacts) > 3:
        parts.append(f"+{len(artifacts) - 3} more")
    return "; ".join(parts) if parts else "-"


def _checkpoint_next_action(run_dir: Path, checkpoint_id: str, checkpoint: dict[str, Any]) -> str | None:
    if checkpoint.get("status") != "needs_review" or not checkpoint.get("approval_required"):
        return None
    approval_path = _checkpoint_approval_path(run_dir, checkpoint_id, checkpoint)
    return (
        "python3 /workspace/scripts/agent_core_cli.py "
        f"--approve-checkpoint {run_dir} {checkpoint_id} "
        '--approved-by "human" --approval-note "reviewed plan/prompts"'
        f"  # writes {approval_path}; executor resume is future work"
    )


def render_checkpoint_summary(run_dir: Path, *, verbose: bool = False) -> bool:
    payload, source = _load_checkpoint_payload(run_dir)
    checkpoints = payload.get("checkpoints") if isinstance(payload.get("checkpoints"), dict) else {}
    if not checkpoints:
        return False
    print("CHECKPOINTS")
    print(_line("Source", source, width=20))
    print(_line("Pipeline", payload.get("pipeline_id"), width=20))
    print(_line("Current checkpoint", payload.get("current_checkpoint_id"), width=20))
    print(_line("Blocked by", payload.get("blocked_by_checkpoint_id"), width=20))
    print()
    for checkpoint_id, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict):
            continue
        status = checkpoint.get("status")
        print(f"  {format_status_icon(status)} {checkpoint_id}")
        print(_line("stage", checkpoint.get("stage"), width=22))
        print(_line("status", status, width=22))
        print(_line("blocking", "yes" if checkpoint.get("blocking") else "no", width=22))
        print(_line("approval_required", "yes" if checkpoint.get("approval_required") else "no", width=22))
        print(_line("reason", short_prompt(checkpoint.get("reason"), 110), width=22))
        print(_line("issues", _compact_list(checkpoint.get("issues")), width=22))
        print(_line("warnings", _compact_list(checkpoint.get("warnings")), width=22))
        print(_line("related_artifacts", _checkpoint_artifact_text(checkpoint.get("related_artifacts")), width=22))
        if checkpoint.get("approval_required") or checkpoint.get("status") == "needs_review" or verbose:
            approval_path = _checkpoint_approval_path(run_dir, str(checkpoint_id), checkpoint)
            print(_line("approval_file", approval_path, width=22))
            if approval_path.exists():
                decision = load_json_safe(approval_path)
                print(_line("approval_decision", f"approved={decision.get('approved')} by={decision.get('approved_by')}", width=22))
        next_action = _checkpoint_next_action(run_dir, str(checkpoint_id), checkpoint)
        if next_action:
            print(_line("next_action", next_action, width=22))
        print()
    blocked = payload.get("blocked_by_checkpoint_id")
    if blocked:
        checkpoint = checkpoints.get(blocked) if isinstance(checkpoints.get(blocked), dict) else {}
        missing = _checkpoint_approval_path(run_dir, str(blocked), checkpoint)
        resume_contract = inspect_resume_contract(run_dir)
        print("RESUME")
        print(_line("Blocked", f"yes · {blocked}", width=18))
        print(_line("Missing file", missing, width=18))
        print(_line("Rejected", "yes" if resume_contract.get("has_rejection") else "no", width=18))
        print(_line("Approve", f"python3 /workspace/scripts/agent_core_cli.py --approve-checkpoint {run_dir} {blocked} --approved-by \"human\" --approval-note \"...\"", width=18))
        print(_line("Resume", "prepared, but executor resume is future work; rerun/start behavior must be defined next", width=18))
        print()
    return True


def _inspect_checkpoints(path: Path, *, verbose: bool = False) -> int:
    run_dir = path
    payload, _source = _load_checkpoint_payload(run_dir)
    if not payload:
        print(f"ERROR: no checkpoints.json/state.json checkpoints found under {run_dir}", file=sys.stderr)
        return 1
    print_box_header("INSPECT CHECKPOINTS", [("Run", run_dir)])
    print()
    render_checkpoint_summary(run_dir, verbose=verbose)
    return 0


def write_checkpoint_decision(
    run_dir: Path,
    checkpoint_id: str,
    *,
    approved: bool,
    actor: str,
    note: str = "",
    force: bool = False,
) -> Path:
    payload, _source = _load_checkpoint_payload(run_dir)
    checkpoints = payload.get("checkpoints") if isinstance(payload.get("checkpoints"), dict) else {}
    if checkpoint_id not in checkpoints:
        raise RuntimeError(f"Checkpoint {checkpoint_id!r} does not exist under {run_dir}")
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")
    approval_path = run_dir / "approvals" / f"{checkpoint_id}.json"
    if not _path_is_relative_to(approval_path, run_dir):
        raise RuntimeError(f"Refusing to write outside run directory: {approval_path}")
    if approval_path.exists() and not force:
        raise RuntimeError(f"Approval file already exists: {approval_path}. Use --force-approval to overwrite.")
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    payload_to_write = {
        "approved": bool(approved),
        "approved_by": actor,
        "approved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
    }
    approval_path.write_text(json.dumps(payload_to_write, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return approval_path


def _write_checkpoint_decision_from_cli(
    values: list[str],
    *,
    approved: bool,
    actor: str,
    note: str,
    force: bool,
) -> int:
    run_dir = _resolve_inspect_run(values[0])
    checkpoint_id = values[1]
    path = write_checkpoint_decision(
        run_dir,
        checkpoint_id,
        approved=approved,
        actor=actor,
        note=note,
        force=force,
    )
    print("CHECKPOINT DECISION WRITTEN")
    print(_line("Run", run_dir, width=14))
    print(_line("Checkpoint", checkpoint_id, width=14))
    print(_line("Approved", str(approved).lower(), width=14))
    print(_line("File", path, width=14))
    print(_line("Resume", "prepared, but executor resume is future work", width=14))
    return 0


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
    duration = job.get("duration_sec")
    duration_text = f"{duration:g}s" if isinstance(duration, (int, float)) else "auto"
    format_text = f"{job.get('orientation')} {job.get('resolution')} · {duration_text}"
    if metadata.get("fps"):
        format_text += f" · {metadata.get('fps')}fps"
    mode_parts = []
    if job.get("use_storyboard"):
        mode_parts.append("Storyboard")
    if job.get("use_voice"):
        mode_parts.append("Voice")
    provider = metadata.get("vision_review_provider")
    if metadata.get("vision_review_enabled") or provider:
        mode_parts.append(f"{provider or 'Vision'} Review")
    if not mode_parts:
        mode_parts.append("Video")
    prompt = job.get("idea") or job.get("script") or job.get("style")
    print_box_header(
        "CONTENT MASCHINE RUN",
        [
            ("Job", job.get("job_id")),
            ("Format", format_text),
            ("Mode", " + ".join(mode_parts)),
            ("Prompt", short_prompt(prompt, 46)),
            ("Started", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ],
    )
    print()
    print("SYSTEM / MODE")
    print(_line("API", f"{_plain_icon('enabled')} {base_url}"))
    print(_line("Director", "pending · local Qwen3.6 if available"))
    print(_line("Voice", f"{_plain_icon(job.get('use_voice'))} {'enabled' if job.get('use_voice') else 'disabled'} · {job.get('voice_id', 'none')}"))
    print(_line("Storyboard", f"{_plain_icon(job.get('use_storyboard'))} {'enabled' if job.get('use_storyboard') else 'disabled'}"))
    print(_line("Video", f"{job.get('pipeline_preference', 'auto')}"))
    print(_line("Vision Review", f"{metadata.get('vision_review_provider', 'default')} · enabled={metadata.get('vision_review_enabled', 'default')}"))
    print(_line("Music", f"{_plain_icon(job.get('use_music'))} {'enabled' if job.get('use_music') else 'disabled'}"))
    print(_line("Subtitles", metadata.get("subtitle_mode", "off")))


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


def summarize_system_mode(payload: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any], base_url: str | None = None) -> list[tuple[str, str]]:
    steps = state.get("steps") or {}
    director = _extract_director_summary(result, state, takes)
    director_bits = []
    if director.get("director_mode"):
        director_bits.append(str(director.get("director_mode")))
    if director.get("director_llm_model"):
        director_bits.append(str(director.get("director_llm_model")))
    elif director.get("director_llm_active") is not None:
        director_bits.append("llm active" if director.get("director_llm_active") else "fallback")

    voice = steps.get("voice") or {}
    storyboard = steps.get("storyboard") or {}
    video = steps.get("video") or {}
    vision_icon, vision_text = _vision_review_status(result)
    return [
        ("API", f"{_plain_icon('enabled')} {base_url or DEFAULT_BASE_URL}"),
        ("Director", f"{_plain_icon('enabled' if director_bits else 'pending')} {' · '.join(director_bits) if director_bits else 'pending'}"),
        ("Voice", f"{_plain_icon(voice.get('status', 'pending'))} {voice.get('status', 'pending')} · {voice.get('backend_name', 'qwen_tts')}"),
        ("Storyboard", f"{_plain_icon(storyboard.get('status', 'pending'))} {storyboard.get('status', 'pending')} · {storyboard.get('backend_name', 'zimage_storyboard')}"),
        ("Video Backend", f"{_plain_icon(video.get('status', 'pending'))} {video.get('backend_name') or 'LTX-2.3'}"),
        ("Vision Review", f"{vision_icon} {vision_text}"),
    ]


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


def _all_scene_outputs(takes: dict[str, Any], state: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for scene in _iter_scene_outputs(takes, state, result):
        key = scene.get("scene_id") or len(scenes)
        if key in seen:
            continue
        seen.add(key)
        scenes.append(scene)
    return scenes


def _review_status(take: dict[str, Any]) -> str:
    review = take.get("take_visual_review") or take.get("metadata", {}).get("take_visual_review") or {}
    return str(take.get("take_visual_review_status") or take.get("review_status") or review.get("take_visual_review_status") or take.get("status") or "pending")


def extract_scene_progress(state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    scenes = _all_scene_outputs(takes, state, result)
    total_scenes = len(scenes) or None
    total_takes = 0
    done_takes = 0
    running_scene = None
    running_take = None
    selected = 0
    counts = {"passed": 0, "needs_review": 0, "rejected": 0, "failed": 0, "pending": 0}
    for index, scene in enumerate(scenes, start=1):
        if scene.get("selected_take_id"):
            selected += 1
        for take_index, take in enumerate(scene.get("takes") or [], start=1):
            total_takes += 1
            status = str(take.get("status") or "pending")
            review_status = _review_status(take)
            bucket = review_status if review_status in counts else status
            if bucket not in counts:
                bucket = "pending"
            counts[bucket] += 1
            if status in {"succeeded", "completed", "failed", "rejected"} or review_status in {"passed", "needs_review", "rejected", "failed"}:
                done_takes += 1
            if running_take is None and status in {"running", "processing", "submitted"}:
                running_scene = index
                running_take = take_index
    return {
        "scene_count": total_scenes,
        "take_count": total_takes or None,
        "done_takes": done_takes,
        "running_scene": running_scene,
        "running_take": running_take,
        "selected": selected,
        "counts": counts,
        "scenes": scenes,
    }


def extract_current_step(status: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("steps") or {}
    phase = status.get("current_phase") or state.get("current_phase") or result.get("final_phase") or "unknown"
    step = steps.get(str(phase)) if isinstance(steps.get(str(phase)), dict) else {}
    if not step:
        for name in ("voice", "storyboard", "music", "video"):
            candidate = steps.get(name)
            if isinstance(candidate, dict) and candidate.get("status") in {"running", "processing", "submitted"}:
                phase = name
                step = candidate
                break
    progress = extract_scene_progress(state, takes, result)
    return {
        "step": phase,
        "status": status.get("status") or step.get("status") or state.get("status"),
        "scene": progress.get("running_scene"),
        "scene_count": progress.get("scene_count"),
        "take": progress.get("running_take"),
        "take_count": progress.get("take_count"),
        "backend": step.get("backend_name") or ("LTX-2.3" if phase == "video" else None),
        "backend_job_id": step.get("backend_job_id"),
        "mode": (step.get("details") or {}).get("mode") or (step.get("params") or {}).get("mode"),
        "prompt": _first_present((step.get("params") or {}).get("prompt"), status.get("status_summary"), result.get("message")),
    }


def render_progress_block(status: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any], *, elapsed: float | None = None, quiet: bool = False) -> None:
    steps = state.get("steps") or {}
    progress = extract_scene_progress(state, takes, result)
    print("PROGRESS")
    validate_status = "succeeded" if state or result else status.get("status", "pending")
    print(f"  {format_status_icon(validate_status)} Validate job")
    director = _extract_director_summary(result, state, takes)
    director_detail = director.get("director_mode") or "pending"
    print(f"  {format_status_icon('succeeded' if director else 'pending')} Director plan        {director_detail}")
    for name, label in (("voice", "Voice"), ("storyboard", "Storyboard"), ("music", "Music"), ("video", "Video render")):
        step = steps.get(name) if isinstance(steps.get(name), dict) else {}
        if not step and name == "music":
            continue
        detail = []
        if step.get("duration_sec") is not None:
            detail.append(f"{step.get('duration_sec')}s")
        if step.get("backend_name"):
            detail.append(str(step.get("backend_name")))
        if name == "video" and progress.get("scene_count"):
            if progress.get("running_scene"):
                detail.append(f"scene {progress.get('running_scene')}/{progress.get('scene_count')}")
            if progress.get("running_take") and progress.get("take_count"):
                detail.append(f"take {progress.get('running_take')}/{progress.get('take_count')}")
        if elapsed is not None and step.get("status") in {"running", "processing", "submitted"}:
            detail.append(f"{format_elapsed(elapsed)} elapsed")
        print(f"  {format_status_icon(step.get('status', 'pending'))} {label:<20} {' · '.join(detail) if detail else step.get('status', 'pending')}")
    if quiet:
        return
    print()
    current = extract_current_step(status, state, takes, result)
    print("CURRENT")
    print(_line("Step", current.get("step")))
    scene_text = f"{current.get('scene')}/{current.get('scene_count')}" if current.get("scene") else "-"
    take_text = f"{current.get('take')}/{current.get('take_count')}" if current.get("take") else "-"
    print(_line("Scene", scene_text))
    print(_line("Take", take_text))
    print(_line("Backend", current.get("backend")))
    print(_line("Mode", current.get("mode") or "running"))
    print(_line("Prompt", short_prompt(current.get("prompt"), 84)))
    checkpoint_id = state.get("current_checkpoint_id")
    blocked_id = state.get("blocked_by_checkpoint_id")
    checkpoints = state.get("checkpoints") or {}
    checkpoint = checkpoints.get(blocked_id or checkpoint_id) if isinstance(checkpoints, dict) else {}
    if checkpoint_id or blocked_id:
        print()
        print("CHECKPOINT")
        print(_line("Current", checkpoint_id))
        print(_line("Blocked by", blocked_id or "-"))
        print(_line("Status", checkpoint.get("status") if isinstance(checkpoint, dict) else "-"))
        print(_line("Approval", "required" if isinstance(checkpoint, dict) and checkpoint.get("approval_required") else "not required"))


def render_scene_summary(state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any], *, verbose: bool = False) -> None:
    scenes = _all_scene_outputs(takes, state, result)
    if not scenes:
        return
    print("SCENES")
    for index, scene in enumerate(scenes, start=1):
        title = scene.get("title") or scene.get("scene_title") or scene.get("scene_id") or f"Scene {index}"
        selected = scene.get("selected_take_id")
        selection = scene.get("selection") or {}
        suffix = f" · selected {selected}" if selected else ""
        if selection.get("technical_selection_status"):
            suffix += f" · {selection.get('technical_selection_status')}"
        print(f"  Scene {index} · {title}{suffix}")
        keyframe = scene.get("selected_keyframe") or scene.get("keyframe") or {}
        key_status = "passed" if keyframe else scene.get("keyframe_status", "waiting")
        print(f"    Keyframe     {_plain_icon(key_status)} {key_status}")
        for take_index, take in enumerate((scene.get("takes") or [])[: 8 if verbose else 4], start=1):
            take_id = take.get("take_id") or f"take {take_index}"
            status = _review_status(take)
            score = take.get("postability_score") or take.get("metadata", {}).get("postability_score")
            score_text = f" · score {score}" if score is not None else ""
            selected_text = " · selected" if selected and take_id == selected else ""
            print(f"    Take {take_index:<2}     {_plain_icon(status)} {status}{selected_text}{score_text}")
    print()


def _load_prompt_trace(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    trace = load_json_safe(run_dir / "model_prompts.json")
    if trace:
        return trace
    return load_json_safe(run_dir / "prompt_audit.json")


def _scene_trace_for_index(trace: dict[str, Any], scene_index: Any) -> dict[str, Any]:
    scenes = trace.get("scenes") or []
    try:
        index = int(scene_index)
    except (TypeError, ValueError):
        index = 1
    if 1 <= index <= len(scenes) and isinstance(scenes[index - 1], dict):
        return scenes[index - 1]
    for scene in scenes:
        if isinstance(scene, dict) and scene.get("scene_id") == f"scene_{index:02d}":
            return scene
    return scenes[0] if scenes and isinstance(scenes[0], dict) else {}


def _live_pipeline_status(state: dict[str, Any], result: dict[str, Any], status_payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    steps = state.get("steps") or {}
    director = _extract_director_summary(result, state, {})
    phase = str(status_payload.get("current_phase") or state.get("current_phase") or "")
    rows = [
        ("Validate", "succeeded" if state or result else status_payload.get("status", "pending"), "job accepted"),
        ("Director plan", "succeeded" if director else ("running" if phase == "planned" else "pending"), director.get("director_mode") or "pending"),
    ]
    for key, label in (("storyboard", "Storyboard"), ("video", "Video render")):
        step = steps.get(key) if isinstance(steps.get(key), dict) else {}
        rows.append((label, step.get("status", "pending"), step.get("backend_name") or ("waiting" if not step else "")))
    rows.append(("Vision review", "pending", "waiting" if not result else "see quality"))
    rows.append(("Assembly", "succeeded" if result.get("success") else "pending", "final summary" if result else "waiting"))
    return rows


def _latest_storyboard_hint(run_dir: Path | None) -> str:
    if not run_dir:
        return "unknown"
    log_path = run_dir / "logs" / "agent.log"
    tail = tail_file(log_path, 80)
    latest = ""
    for line in tail.splitlines():
        if "storyboard candidate" in line:
            latest = line
    if not latest:
        return "unknown"
    return _short_text(latest, 70)


def _format_live_lines(
    payload: dict[str, Any],
    state: dict[str, Any],
    takes: dict[str, Any],
    result: dict[str, Any],
    *,
    run_dir: Path | None,
    base_url: str,
    start: float,
    phase_started: float,
    quiet: bool,
    verbose: bool,
) -> list[str]:
    now = time.monotonic()
    current = extract_current_step(payload, state, takes, result)
    trace = _load_prompt_trace(run_dir)
    scene_trace = _scene_trace_for_index(trace, current.get("scene") or 1)
    job_id = payload.get("job_id") or state.get("job_id") or result.get("job_id") or "-"
    mode_id = trace.get("mode_id") or result.get("metadata", {}).get("mode_id") or "-"
    style_id = trace.get("style_id") or result.get("metadata", {}).get("style_id") or "-"
    phase = payload.get("current_phase") or state.get("current_phase") or current.get("step") or "unknown"
    status = payload.get("status") or state.get("status") or "unknown"
    width = 60
    header_rows = [
        ("Job", job_id),
        ("Mode", f"{mode_id} · {style_id}" if mode_id != "-" or style_id != "-" else "-"),
        ("Format", "-"),
        ("Status", f"{status} · {phase} · elapsed {format_elapsed(now - start)}"),
    ]
    if result.get("metadata"):
        meta = result.get("metadata") or {}
        if meta.get("width") and meta.get("height"):
            header_rows[2] = ("Format", f"{meta.get('orientation', '-')} {meta.get('width')}x{meta.get('height')}")
    lines = ["╭" + "─" * width + "╮", "│ CONTENT MASCHINE LIVE".ljust(width + 1) + "│", "├" + "─" * width + "┤"]
    for label, value in header_rows:
        body = f"{label:<10} {str(value or '-')}"
        lines.append("│ " + body[: width - 1].ljust(width - 1) + "│")
    lines.append("╰" + "─" * width + "╯")
    lines.append("")
    lines.append("SYSTEM")
    for label, value in summarize_system_mode({}, state, takes, result, base_url):
        lines.append(_line(label, value, width=14))
    subtitle_mode = result.get("metadata", {}).get("subtitle_mode") or "off"
    lines.append(_line("Subtitles", subtitle_mode, width=14))
    lines.append("")
    lines.append("PIPELINE")
    for label, step_status, detail in _live_pipeline_status(state, result, payload):
        lines.append(f"  {format_status_icon(step_status)} {label:<18} {detail or step_status}")
    checkpoint_id = state.get("current_checkpoint_id")
    blocked_id = state.get("blocked_by_checkpoint_id")
    checkpoints = state.get("checkpoints") or {}
    checkpoint = checkpoints.get(blocked_id or checkpoint_id) if isinstance(checkpoints, dict) else {}
    if checkpoint_id or blocked_id:
        lines.append("")
        lines.append("CHECKPOINT")
        lines.append(_line("Current", checkpoint_id or "-", width=14))
        lines.append(_line("Blocked", blocked_id or "no", width=14))
        if isinstance(checkpoint, dict):
            lines.append(_line("Status", checkpoint.get("status") or "-", width=14))
            lines.append(_line("Approval", "required" if checkpoint.get("approval_required") else "not required", width=14))
            if blocked_id and run_dir:
                lines.append(_line("Next action", f"approve {blocked_id} via --approve-checkpoint", width=14))
    if quiet:
        return lines
    lines.append("")
    lines.append("CURRENT WORK")
    scene_text = f"scene_{int(current['scene']):02d}" if current.get("scene") else "unknown"
    if scene_trace:
        scene_text += f" · {scene_trace.get('scene_role') or scene_trace.get('role') or '-'} · {scene_trace.get('motif_id') or '-'}"
    take_text = f"{current.get('take')}/{current.get('take_count')}" if current.get("take") else "-"
    phase_elapsed = now - phase_started
    lines.append(_line("Step", current.get("step"), width=14))
    lines.append(_line("Scene", scene_text, width=14))
    lines.append(_line("Candidate", _latest_storyboard_hint(run_dir) if str(current.get("step")) == "storyboard" else take_text, width=14))
    lines.append(_line("Backend", current.get("backend") or "unknown", width=14))
    lines.append(_line("Elapsed", format_elapsed(phase_elapsed), width=14))
    lines.append(_line("Best ETA", "unknown", width=14))
    if phase_elapsed >= 600:
        lines.append(_line("Note", "check backend log if no progress", width=14))
    elif phase_elapsed >= 300:
        lines.append(_line("Note", "long-running but process not known failed", width=14))
    lines.append("")
    lines.append("CURRENT PROMPT")
    positive = scene_trace.get("positive_model_prompt") or "-"
    negative = scene_trace.get("negative_model_prompt") or "-"
    zimage_prompt = scene_trace.get("zimage_prompt_sent")
    ltx_prompt = scene_trace.get("ltx_prompt_sent")
    policy = (trace.get("backend_prompt_policy") or scene_trace.get("backend_prompt_policy") or {})
    if str(current.get("step")) == "storyboard" and zimage_prompt:
        actual = zimage_prompt
    elif str(current.get("step")) == "video" and ltx_prompt:
        actual = ltx_prompt
    else:
        actual = zimage_prompt or ltx_prompt or positive
    lines.append(_line("Positive", short_prompt(positive, 96), width=14))
    lines.append(_line("Negative", short_prompt(negative, 96), width=14))
    lines.append(_line("Actual", short_prompt(actual, 96), width=14))
    lines.append(_line("Policy", f"zimage {policy.get('zimage', 'unknown')} / ltx {policy.get('ltx', 'unknown')}" if isinstance(policy, dict) else "unknown", width=14))
    lines.append("")
    lines.append("SCENES")
    trace_scenes = trace.get("scenes") or []
    if trace_scenes:
        for idx, scene in enumerate(trace_scenes[:6], start=1):
            marker = "[>]" if current.get("scene") == idx else "[-]"
            if result.get("success"):
                marker = "[✓]"
            lines.append(
                f"  Scene {idx:<2} {marker} {scene.get('scene_role') or scene.get('role') or '-'} · {scene.get('shot_recipe_id') or scene.get('motif_id') or '-'}"
            )
    else:
        lines.append("  pending")
    lines.append("")
    lines.append("ARTIFACTS")
    lines.append(_line("Run folder", run_dir or "pending", width=14))
    if run_dir:
        lines.append(_line("Prompt audit", "ready" if (run_dir / "prompt_audit.json").exists() else "pending", width=14))
        lines.append(_line("Model prompts", "ready" if (run_dir / "model_prompts.json").exists() else "pending", width=14))
    if verbose and payload.get("status_summary"):
        lines.append(_line("Summary", short_prompt(payload.get("status_summary"), 110), width=14))
    return lines


def render_live_dashboard(
    payload: dict[str, Any],
    state: dict[str, Any],
    takes: dict[str, Any],
    result: dict[str, Any],
    *,
    run_dir: Path | None,
    base_url: str,
    start: float,
    phase_started: float,
    quiet: bool,
    verbose: bool,
) -> None:
    lines = _format_live_lines(
        payload,
        state,
        takes,
        result,
        run_dir=run_dir,
        base_url=base_url,
        start=start,
        phase_started=phase_started,
        quiet=quiet,
        verbose=verbose,
    )
    sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


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


def _real_vlm_from_verdict(verdict: dict[str, Any]) -> bool | None:
    frame_review = verdict.get("final_frame_review") or {}
    if not isinstance(frame_review, dict) or not frame_review:
        return None
    if frame_review.get("real_vlm_inference_used") is not None:
        return bool(frame_review.get("real_vlm_inference_used"))
    provider = frame_review.get("provider")
    warnings = [str(item).lower() for item in frame_review.get("warnings") or []]
    return provider == "qwen3_vl" and not any(
        marker in warning
        for warning in warnings
        for marker in ("skipped", "unavailable", "failed", "model dir not ready", "not recognized")
    )


def _all_quality_messages(verdict: dict[str, Any]) -> list[str]:
    return [str(item) for item in list(verdict.get("main_issues") or []) + list(verdict.get("warnings") or [])]


def _quality_message_category(message: str) -> str:
    text = message.lower()
    if "subtitle" in text or "burned subtitles" in text:
        return "policy"
    if "qwen3_vl inference failed" in text or "python not found" in text or "model type" in text or "runtime missing" in text:
        return "vision_runtime"
    if "non-json" in text or "json" in text or "parser" in text:
        return "vision_review"
    return "quality"


def _group_quality_messages(verdict: dict[str, Any]) -> dict[str, list[str]]:
    groups = {"quality": [], "vision_review": [], "vision_runtime": [], "policy": []}
    for message in _all_quality_messages(verdict):
        groups[_quality_message_category(message)].append(message)
    return groups


def _vision_review_status(result: dict[str, Any]) -> tuple[str, str]:
    verdict = _extract_quality_verdict(result)
    frame_review = verdict.get("final_frame_review") if isinstance(verdict, dict) else {}
    provider = _first_present(
        (frame_review or {}).get("provider") if isinstance(frame_review, dict) else None,
        verdict.get("visual_review_provider") if isinstance(verdict, dict) else None,
        result.get("metadata", {}).get("vision_review_provider") if isinstance(result.get("metadata"), dict) else None,
        "heuristic",
    )
    messages = " ".join(_all_quality_messages(verdict)).lower() if verdict else ""
    real_vlm = _real_vlm_from_verdict(verdict) if verdict else None
    if provider == "qwen3_vl" and ("python not found" in messages or "model type" in messages or "runtime missing" in messages or "inference failed" in messages):
        return "✗", "qwen3_vl · runtime missing"
    if provider == "qwen3_vl" and ("non-json" in messages or "parser" in messages):
        return "⚠", "qwen3_vl · parser warning"
    if provider == "qwen3_vl" and real_vlm is True:
        return "✓", "qwen3_vl · real inference used"
    if provider == "qwen3_vl" and real_vlm is False:
        return "⚠", "qwen3_vl · no real inference"
    if provider and provider != "heuristic":
        return "-", str(provider)
    return "-", "heuristic"


def _selected_take_for_scene(scene: dict[str, Any]) -> dict[str, Any]:
    selected = scene.get("selected_take_id")
    for take in scene.get("takes") or []:
        if take.get("take_id") == selected:
            return take
    return {}


def _take_review(take: dict[str, Any]) -> dict[str, Any]:
    review = take.get("take_visual_review") or take.get("metadata", {}).get("take_visual_review") or {}
    return review if isinstance(review, dict) else {}


def _format_score(score: Any) -> str:
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _scene_icon(status: str, score: Any, marker: str) -> str:
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None
    lowered = f"{status} {marker}".lower()
    if "rejected" in lowered or "failed" in lowered or (score_value is not None and score_value < 0.4):
        return "✗"
    if "needs_review" in lowered or "warning" in lowered or "parser" in lowered or (score_value is not None and score_value < 0.8):
        return "⚠"
    return "✓"


def _next_action_hints(verdict: dict[str, Any]) -> list[str]:
    messages = " ".join(_all_quality_messages(verdict)).lower()
    hints: list[str] = []
    if "burned subtitles" in messages or "subtitle" in messages:
        hints.append("Suggested rerun for clean visual test: use --subtitle-mode off or sidecar.")
    if "non-json" in messages or "parser" in messages:
        hints.append("Inspect qwen3_vl review warnings in takes.json; if frequent, improve subprocess JSON extraction later.")
    if "final frame review rejected" in messages:
        hints.append("Open final.mp4 and inspect the rejected scene/frame before changing prompts.")
    if "selected take needs visual review" in messages:
        hints.append("Consider --takes-per-scene 3 or tune the scene prompt/motif.")
    deduped: list[str] = []
    for hint in hints:
        if hint not in deduped:
            deduped.append(hint)
    return deduped


def render_quality_summary(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], *, live: bool = False, verbose: bool = False) -> None:
    verdict = _extract_quality_verdict(result)
    progress = extract_scene_progress(state, takes, result)
    counts = progress.get("counts") or {}
    if live:
        print("QUALITY LIVE")
        print(_line("Selected takes", f"{counts.get('passed', 0)} passed · {counts.get('needs_review', 0)} review · {counts.get('rejected', 0) + counts.get('failed', 0)} rejected"))
        provider = "pending"
        if verdict:
            frame_review = verdict.get("final_frame_review") or {}
            provider = frame_review.get("provider") if isinstance(frame_review, dict) else provider
        print(_line("Vision provider", provider or "pending"))
        real_vlm = _real_vlm_from_verdict(verdict) if verdict else None
        print(_line("Real VLM", "pending" if real_vlm is None else real_vlm))
        print()
        return
    if not verdict:
        return
    print("QUALITY VERDICT")
    print(_line("Status", verdict.get("final_quality_status", "unknown")))
    print(_line("Score", verdict.get("final_postability_score", "unknown")))
    real_vlm = _real_vlm_from_verdict(verdict)
    print(_line("Real VLM", f"✓ {real_vlm}" if real_vlm is True else ("⚠ False" if real_vlm is False else "unknown")))
    vision_icon, vision_text = _vision_review_status(result)
    provider = vision_text.split(" · ", 1)[0] if vision_text else "unknown"
    print(_line("Provider", f"{vision_icon} {provider}"))
    print(_line("Recommendation", verdict.get("recommended_next_action", "unknown")))
    groups = _group_quality_messages(verdict)
    section_specs = [
        ("QUALITY ISSUES", groups["quality"], None),
        ("VISION RUNTIME WARNINGS", groups["vision_runtime"], None),
        ("VISION REVIEW WARNINGS", groups["vision_review"], None),
        ("POLICY / CONFIG WARNINGS", groups["policy"], "subtitle-mode=burn intentionally adds visible text. Use --subtitle-mode off or sidecar for clean no-text visual tests."),
    ]
    limit = 12 if verbose else 6
    for title, messages, hint in section_specs:
        if not messages:
            continue
        print()
        print(title)
        for message in messages[:limit]:
            print(f"  ! {short_prompt(message, 116)}")
        if hint:
            print(f"    hint: {hint}")
    print()


def _scene_result_lines(state: dict[str, Any], takes: dict[str, Any], result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for index, scene in enumerate(_all_scene_outputs(takes, state, result), start=1):
        selected = scene.get("selected_take_id") or "none"
        selection = scene.get("selection") or {}
        selected_take = _selected_take_for_scene(scene)
        review = _take_review(selected_take)
        scene_status = (
            selection.get("visual_selection_status")
            or review.get("take_visual_review_status")
            or selection.get("technical_selection_status")
            or ("passed" if selected != "none" else "unknown")
        )
        score = selection.get("postability_score") or selection.get("selected_postability_score") or review.get("postability_score")
        if score is None:
            for take in scene.get("takes") or []:
                if take.get("take_id") == selected:
                    score = take.get("postability_score") or take.get("metadata", {}).get("postability_score")
        provider = review.get("provider") or selected_take.get("visual_review_provider") or selected_take.get("metadata", {}).get("visual_review_provider")
        warnings = " ".join(str(item) for item in review.get("warnings") or [])
        issues = " ".join(str(item) for item in review.get("issues") or [])
        marker = ""
        if "non-json" in warnings.lower() or "parser" in warnings.lower():
            marker = "parser warning"
        elif "inference failed" in warnings.lower() or "python not found" in warnings.lower() or "model type" in warnings.lower():
            marker = "runtime warning"
        elif warnings or issues:
            marker = "warning"
        icon = _scene_icon(str(scene_status), score, marker)
        score_text = _format_score(score)
        provider_text = f" · {provider}" if provider else ""
        marker_text = f" {marker}" if marker else ""
        lines.append(f"  Scene {index:<2} {icon} {str(scene_status):<12} · take {selected} · score {score_text}{provider_text}{marker_text}")
    return lines


def render_success_dashboard(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], run_dir: Path | None, base_url: str, *, verbose: bool = False) -> None:
    print_box_header("RUN COMPLETE")
    final_path = result.get("output_final_path") or _artifact_path(result, "final_output_mp4")
    print()
    print("RESULT")
    print(_line("Status", "✓ success"))
    print(_line("Final phase", result.get("final_phase")))
    print(_line("Final video", final_path))
    print(_line("Size", format_file_size(final_path)))
    print(_line("Duration", f"{result.get('actual_final_duration_sec')}s" if result.get("actual_final_duration_sec") is not None else "unknown"))
    print()
    print("PIPELINE")
    for label, value in summarize_system_mode({}, state, takes, result, base_url):
        print(_line(label, value))
    progress = extract_scene_progress(state, takes, result)
    print(_line("Render", f"✓ {progress.get('scene_count') or 0} scenes · {progress.get('take_count') or 0} takes"))
    print(_line("Assembly", "✓ final.mp4 created" if final_path else "unknown"))
    print()
    if run_dir:
        render_checkpoint_summary(run_dir, verbose=verbose)
    render_quality_summary(result, state, takes, live=False, verbose=verbose)
    scene_lines = _scene_result_lines(state, takes, result)
    if scene_lines:
        print("SCENE SUMMARY")
        for line in scene_lines:
            print(line)
        print()
    print("NEXT ACTION")
    action_index = 1
    if final_path:
        print(f"  {action_index}. Open video:")
        print(f"     {final_path}")
        action_index += 1
    if result.get("job_id"):
        print(f"  {action_index}. Inspect run:")
        print(f"     python3 /workspace/scripts/agent_core_cli.py --inspect-run {result.get('job_id')}")
        action_index += 1
    verdict = _extract_quality_verdict(result)
    for hint in _next_action_hints(verdict):
        print(f"  {action_index}. {hint}")
        action_index += 1
    if run_dir:
        print()
        print("ARTIFACTS")
        print(_line("Run folder", run_dir))
        log_path = run_dir / "logs" / "agent.log"
        if log_path.exists():
            print(_line("Logs", log_path))


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


def extract_root_cause_from_log_tail(log_tail: str) -> dict[str, str]:
    patterns = (
        "CUDA out of memory",
        "FileNotFoundError",
        "AttributeError",
        "RuntimeError",
        "ImportError",
        "ModuleNotFoundError",
    )
    lines = [line.rstrip() for line in (log_tail or "").splitlines() if line.strip()]
    cause = ""
    for line in reversed(lines):
        if any(pattern in line for pattern in patterns):
            cause = line.strip()
            break
    if not cause:
        return {"root_cause": "unknown; inspect backend log", "likely_meaning": "unknown; inspect backend log"}
    lower = cause.lower()
    meaning = "unknown; inspect backend log"
    if "tokenizer.model" in lower:
        meaning = "Gemma/LTX model folder looks incomplete; tokenizer.model is missing."
    elif "siglipvisionmodel" in lower and "vision_model" in lower:
        meaning = "LTX/Gemma runtime dependency mismatch. Global transformers version may be incompatible with LTX."
    elif "cuda out of memory" in lower:
        meaning = "GPU VRAM exhausted. Reduce resolution, scene count, takes, or concurrent model load."
    elif "qwen3_vl" in lower and ("not recognized" in lower or "unrecognized" in lower):
        meaning = "Wrong Transformers runtime for Qwen3-VL; use the isolated Qwen3-VL review runtime."
    elif "modulenotfounderror" in lower or "importerror" in lower:
        meaning = "Python runtime dependency is missing or loaded from the wrong environment."
    return {"root_cause": cause, "likely_meaning": meaning}


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
    render_success_dashboard(result, state, takes, run_dir, base_url, verbose=verbose)


def print_failure_summary(result: dict[str, Any], state: dict[str, Any], takes: dict[str, Any], run_dir: Path | None, *, tail_lines: int = 80, show_tail: bool = True) -> None:
    details = _extract_failure_details(result, state, takes)
    tail = tail_file(details.get("log_file"), tail_lines) if details.get("log_file") else ""
    diagnosis = extract_root_cause_from_log_tail(tail)
    print_box_header("RUN FAILED")
    print()
    print("ERROR")
    print(_line("Phase", details.get("phase")))
    print(_line("Scene", details.get("scene_id") or "unknown"))
    print(_line("Take", details.get("take_id") or "unknown"))
    print(_line("Backend", details.get("backend") or "unknown"))
    print(_line("Message", short_prompt(details.get("backend_error") or details.get("agent_error"), 120)))
    if details.get("backend_job_id"):
        print(_line("Backend job", details.get("backend_job_id")))
    print()
    print("ROOT CAUSE")
    print(_line("Detected", diagnosis.get("root_cause")))
    print()
    print("LIKELY MEANING")
    print(f"  {diagnosis.get('likely_meaning')}")
    print()
    print("FILES")
    if details.get("log_file"):
        print(_line("Backend log", details.get("log_file")))
    if run_dir:
        print(_line("Result", run_dir / "result.json"))
        print(_line("State", run_dir / "state.json"))
        if (run_dir / "takes.json").exists():
            print(_line("Takes", run_dir / "takes.json"))
    if run_dir:
        print()
        render_checkpoint_summary(run_dir, verbose=False)
    render_quality_summary(result, state, takes, live=False, verbose=False)
    if show_tail and tail:
        print("LOG TAIL")
        print("  " + "─" * 56)
        for line in tail.splitlines()[-tail_lines:]:
            print(f"  {line}")
        print("  " + "─" * 56)
        print()
    if details.get("log_file"):
        print("NEXT DEBUG COMMAND")
        print(f"  cat {details.get('log_file')}")


def _status_signature(payload: dict[str, Any], state: dict[str, Any], takes: dict[str, Any]) -> tuple[Any, ...]:
    steps = state.get("steps") or {}
    step_sig = tuple((name, (steps.get(name) or {}).get("status"), (steps.get(name) or {}).get("backend_job_id")) for name in ("voice", "storyboard", "music", "video"))
    director = _extract_director_summary(payload.get("result") or {}, state, takes)
    checkpoints = state.get("checkpoints") or {}
    checkpoint_sig = tuple(
        (key, value.get("status"), value.get("approval_required"))
        for key, value in checkpoints.items()
        if isinstance(value, dict)
    )
    return (
        payload.get("status"),
        payload.get("current_phase"),
        payload.get("status_summary"),
        state.get("current_checkpoint_id"),
        state.get("blocked_by_checkpoint_id"),
        checkpoint_sig,
        step_sig,
        tuple(_take_lines(takes, state, payload.get("result") or {}, failures_only=False)[:8]),
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
    if args.pipeline_dry_run:
        job["metadata"]["pipeline_dry_run"] = True
    if args.approval_gates_enabled:
        job["metadata"]["approval_gates_enabled"] = True
    if args.stop_after:
        job["metadata"]["stop_after"] = args.stop_after

    return {"job": job}


def _absolute_url(base_url: str, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    return parse.urljoin(f"{base_url}/", maybe_relative.lstrip("/"))


def _print_submit(payload: dict[str, Any], submit_response: dict[str, Any], base_url: str) -> None:
    job_id = submit_response.get("job_id") or payload["job"]["job_id"]
    poll_url = _absolute_url(base_url, submit_response.get("poll_url"))
    print()
    print("SUBMITTED")
    print(_line("Job", job_id))
    if poll_url:
        print(_line("Poll URL", poll_url))
    print(_line("Initial", f"{submit_response.get('status')} · {submit_response.get('current_phase')}"))


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
    live: bool = False,
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
            if live:
                render_live_dashboard(
                    payload,
                    state,
                    takes,
                    result,
                    run_dir=run_dir,
                    base_url=base_url,
                    start=start,
                    phase_started=phase_started,
                    quiet=quiet,
                    verbose=verbose,
                )
            else:
                print()
                print(f"UPDATE {format_elapsed(now - start)} · {payload.get('status')} · phase {phase} · phase elapsed {format_elapsed(now - phase_started)}")
                if payload.get("status_summary"):
                    print(_line("Summary", short_prompt(payload.get("status_summary"), 110)))
                if not quiet:
                    render_progress_block(payload, state, takes, result, elapsed=now - start, quiet=quiet)
                    render_scene_summary(state, takes, result, verbose=verbose)
                    render_quality_summary(result, state, takes, live=True, verbose=verbose)
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

    if payload.get("success") or result.get("success"):
        print_success_summary(result, state, takes, run_dir, base_url, verbose=verbose)
    else:
        print_failure_summary(result, state, takes, run_dir, tail_lines=tail_lines, show_tail=show_log_tail)

    if verbose:
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
    if error_payload and verbose:
        print(f"error: {json.dumps(error_payload, ensure_ascii=True)}")


def _inspect_run(path: Path, *, tail_lines: int, show_log_tail: bool, verbose: bool) -> int:
    run_dir = path
    result = load_json_safe(run_dir / "result.json")
    state = load_json_safe(run_dir / "state.json")
    takes = load_json_safe(run_dir / "takes.json")
    if not result and not state:
        print(f"ERROR: no result.json/state.json found under {run_dir}", file=sys.stderr)
        return 1
    print_box_header("INSPECT RUN", [("Run", run_dir)])
    print()
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
    live = _should_use_live(args)

    try:
        if args.inspect_run:
            return _inspect_run(
                _resolve_inspect_run(args.inspect_run),
                tail_lines=args.tail_error_log_lines,
                show_log_tail=not args.no_log_tail,
                verbose=args.verbose,
            )
        if args.inspect_checkpoints:
            return _inspect_checkpoints(_resolve_inspect_run(args.inspect_checkpoints), verbose=args.verbose)
        if args.approve_checkpoint:
            return _write_checkpoint_decision_from_cli(
                args.approve_checkpoint,
                approved=True,
                actor=args.approved_by,
                note=args.approval_note,
                force=args.force_approval,
            )
        if args.reject_checkpoint:
            return _write_checkpoint_decision_from_cli(
                args.reject_checkpoint,
                approved=False,
                actor=args.rejected_by,
                note=args.approval_note,
                force=args.force_approval,
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
            live=live,
        )
        if live:
            print()
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
