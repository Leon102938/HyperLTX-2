#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
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
    parser.add_argument("--style", default="cinematic", help="Style hint. Default: %(default)s")
    parser.add_argument(
        "--pipeline-preference",
        choices=("auto", "ti2vid", "a2vid", "fast", "balanced", "quality"),
        default="auto",
        help="Preferred render pipeline. Default: %(default)s",
    )
    parser.add_argument("--poll-interval-sec", type=float, default=3.0, help="Fallback polling interval. Default: %(default)s")
    parser.add_argument("--timeout-sec", type=float, default=1800.0, help="Overall wait timeout. Default: %(default)s")
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


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.idea and not args.script:
        raise RuntimeError("At least one of --idea or --script is required.")

    job: dict[str, Any] = {
        "job_id": args.job_id or _generate_job_id(),
        "idea": args.idea,
        "script": args.script,
        "resolution": args.resolution,
        "use_voice": args.use_voice,
        "use_storyboard": args.use_storyboard,
        "style": args.style,
        "pipeline_preference": args.pipeline_preference,
        "orientation": args.orientation,
    }

    if args.duration_sec is not None:
        job["duration_sec"] = args.duration_sec
    if args.use_voice:
        job["voice_id"] = args.voice_id

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


def _poll_job(base_url: str, job_id: str, *, timeout_sec: float, fallback_poll_sec: float) -> dict[str, Any]:
    poll_url = f"{base_url}/agent-core/jobs/{job_id}"
    deadline = time.monotonic() + timeout_sec
    last_signature: tuple[Any, Any, Any] | None = None

    while True:
        payload = _http_json(poll_url, timeout=30.0)
        signature = (payload.get("status"), payload.get("current_phase"), payload.get("status_summary"))
        if signature != last_signature:
            print(
                f"Status: {payload.get('status')} phase={payload.get('current_phase')} "
                f"summary={payload.get('status_summary')}"
            )
            last_signature = signature

        if payload.get("is_terminal"):
            return payload

        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out while waiting for job {job_id} at {poll_url}")

        retry_after = payload.get("retry_after_sec")
        sleep_for = retry_after if isinstance(retry_after, (int, float)) and retry_after > 0 else fallback_poll_sec
        time.sleep(sleep_for)


def _print_terminal(payload: dict[str, Any], base_url: str) -> None:
    result = payload.get("result") or {}
    refs = payload.get("refs") or {}
    public_refs = payload.get("public_refs") or {}
    print(f"Terminal status: {payload.get('status')} success={payload.get('success')}")

    director_mode, director_llm_active = _extract_director_fields(result)
    if director_mode is not None:
        print(f"Director mode: {director_mode}")
    if director_llm_active is not None:
        print(f"Director LLM active: {director_llm_active}")

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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    base_url = _normalize_base_url(args.base_url)

    try:
        payload = _build_payload(args)
        if args.print_payload:
            print(json.dumps(payload, indent=2, ensure_ascii=True))

        submit_response = _http_json(f"{base_url}/agent-core/jobs", method="POST", payload=payload, timeout=30.0)
        _print_submit(payload, submit_response, base_url)

        terminal_payload = _poll_job(
            base_url,
            submit_response.get("job_id") or payload["job"]["job_id"],
            timeout_sec=args.timeout_sec,
            fallback_poll_sec=args.poll_interval_sec,
        )
        _print_terminal(terminal_payload, base_url)
        return 0 if terminal_payload.get("success") else 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
