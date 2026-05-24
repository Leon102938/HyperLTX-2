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


WORKSPACE = Path("/workspace")
HF_HOME = WORKSPACE / ".cache" / "hf"
HF_HUB_CACHE = HF_HOME / "hub"
STATUS_JSON = WORKSPACE / "status" / "hf_xet_benchmark.json"
MARKDOWN_REPORT = WORKSPACE / "HF_XET_BENCHMARK.md"
DEFAULT_REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_FILENAME = "model.safetensors"
MIN_BYTES = 1_000_000_000
MAX_BYTES = 3_500_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def force_env() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)


def preflight() -> dict[str, Any]:
    force_env()
    status: dict[str, Any] = {
        "xet_preflight_ok": False,
        "high_performance_env": os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1",
        "HF_HOME": os.environ.get("HF_HOME"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET"),
        "errors": [],
    }

    def require(condition: bool, message: str) -> None:
        if not condition:
            status["errors"].append(message)

    try:
        import huggingface_hub

        status["huggingface_hub"] = getattr(huggingface_hub, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - fatal environment path
        status["huggingface_hub_error"] = repr(exc)
        require(False, "huggingface_hub import failed")

    try:
        import hf_xet

        status["hf_xet"] = getattr(hf_xet, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - fatal environment path
        status["hf_xet_error"] = repr(exc)
        require(False, "hf_xet import failed")

    require(os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1", "HF_XET_HIGH_PERFORMANCE must be 1")
    require(os.environ.get("HF_HUB_DISABLE_XET") in (None, "", "0", "false", "False"), "HF_HUB_DISABLE_XET must not disable Xet")

    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(key)
        require(bool(value), f"{key} is not set")
        require(str(value).startswith("/workspace/"), f"{key} must point to /workspace")
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            require(os.access(path, os.W_OK), f"{key} is not writable: {value}")

    status["xet_preflight_ok"] = not status["errors"]
    if status["errors"]:
        raise SystemExit("HF/Xet preflight failed: " + "; ".join(status["errors"]))
    return status


def get_file_metadata(repo_id: str, filename: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            size = getattr(sibling, "size", None)
            if size is None:
                raise SystemExit(f"file size unavailable for {repo_id}/{filename}")
            if size < MIN_BYTES:
                raise SystemExit(f"test file too small: {size} bytes < {MIN_BYTES}")
            if size > MAX_BYTES:
                raise SystemExit(f"test file too large: {size} bytes > {MAX_BYTES}")
            return {
                "repo_id": repo_id,
                "filename": filename,
                "bytes": size,
                "gated": getattr(info, "gated", None),
                "private": getattr(info, "private", None),
            }
    raise SystemExit(f"file not found in repo metadata: {repo_id}/{filename}")


def cached_path(repo_id: str, filename: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None


def xet_log_state() -> dict[str, Any]:
    log_dir = HF_HOME / "xet" / "logs"
    files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    latest = files[-1] if files else None
    return {
        "count": len(files),
        "latest": str(latest) if latest else None,
        "latest_mtime": latest.stat().st_mtime if latest else None,
    }


def run_download(repo_id: str, filename: str, metadata: dict[str, Any], preflight_status: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    cached_before = cached_path(repo_id, filename)
    log_before = xet_log_state()
    start = time.perf_counter()
    start_time = utc_now()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(HF_HUB_CACHE),
    )
    end = time.perf_counter()
    end_time = utc_now()
    file_path = Path(path)
    file_bytes = file_path.stat().st_size
    duration = max(end - start, 0.000001)
    cache_hit = cached_before is not None
    log_after = xet_log_state()
    mb_s = (file_bytes / 1_000_000) / duration
    mib_s = (file_bytes / (1024 * 1024)) / duration
    mbps = (file_bytes * 8 / duration) / 1_000_000
    return {
        "repo_id": repo_id,
        "filename": filename,
        "expected_bytes": metadata["bytes"],
        "cache_path": path,
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": round(duration, 3),
        "file_bytes": file_bytes,
        "MB_per_sec": round(mb_s, 3),
        "MiB_per_sec": round(mib_s, 3),
        "Mbps": round(mbps, 3),
        "Gbps": round(mbps / 1000, 3),
        "cache_hit": cache_hit,
        "xet_preflight_ok": preflight_status["xet_preflight_ok"],
        "high_performance_env": preflight_status["high_performance_env"],
        "xet_log_before": log_before,
        "xet_log_after": log_after,
        "xet_log_changed": log_before != log_after,
    }


def load_existing_report() -> dict[str, Any]:
    if STATUS_JSON.exists():
        try:
            data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("runs"), list):
                return data
        except json.JSONDecodeError:
            pass
    return {"runs": []}


def write_reports(payload: dict[str, Any]) -> None:
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = payload["environment"]
    testfile = payload["testfile"]
    runs = payload["runs"]
    miss = next((run for run in runs if not run["cache_hit"]), None)
    hit = next((run for run in reversed(runs) if run["cache_hit"]), None)

    def run_lines(run: dict[str, Any] | None) -> str:
        if not run:
            return "- duration: n/a\n- MB/s: n/a\n- MiB/s: n/a\n- Gbps: n/a\n- cache_hit: n/a"
        return "\n".join(
            [
                f"- duration: {run['duration_sec']} s",
                f"- MB/s: {run['MB_per_sec']}",
                f"- MiB/s: {run['MiB_per_sec']}",
                f"- Gbps: {run['Gbps']}",
                f"- cache_hit: {str(run['cache_hit']).lower()}",
            ]
        )

    miss_speed = miss["Gbps"] if miss else None
    if miss_speed is None:
        speed_eval = "kein Cache-Miss-Run im Report; bitte Cache gezielt entfernen oder neues Testfile waehlen"
    elif miss_speed >= 2.0:
        speed_eval = "HF/Xet liegt nahe am gemessenen Cloudflare/CDN-Pod-Speed"
    elif miss_speed >= 1.0:
        speed_eval = "HF/Xet ist brauchbar, aber klar unter Cloudflare/CDN"
    else:
        speed_eval = "HF/Xet ist deutlich unter Cloudflare/CDN und sollte weiter untersucht werden"

    md = f"""# HF/Xet Benchmark

## 1. Ziel
- Kontrolliert messen, ob Hugging Face/Xet mit High-Performance-Env real in die Naehe des Pod-CDN-Speeds kommt.

## 2. Environment
- huggingface_hub: {env.get('huggingface_hub')}
- hf_xet: {env.get('hf_xet')}
- HF_XET_HIGH_PERFORMANCE: {env.get('high_performance_env')}
- HF_HUB_DISABLE_XET: {env.get('HF_HUB_DISABLE_XET')}
- HF_HOME: {env.get('HF_HOME')}
- Cache: {env.get('HF_HUB_CACHE')}

## 3. Testfile
- repo_id: {testfile['repo_id']}
- filename: {testfile['filename']}
- bytes: {testfile['bytes']}
- oeffentlich/gated: gated={testfile.get('gated')}, private={testfile.get('private')}
- warum geeignet: einzelnes oeffentliches File, per hf_hub_download ladbar, zwischen 1 GB und 3 GB

## 4. Run 1 Cache Miss
{run_lines(miss)}

## 5. Run 2 Cache Hit
{run_lines(hit)}

## 6. Bewertung
- HF/Xet-Speed: {speed_eval}
- Vergleich zu Cloudflare 2.4-2.9 Gbps: {miss_speed if miss_speed is not None else 'n/a'} Gbps im Cache-Miss-Run
- Netzwerk-Bottleneck ja/nein: {'nein' if miss_speed and miss_speed >= 2.0 else 'unklar'}
- HF/Xet-Bottleneck ja/nein: {'nein' if miss_speed and miss_speed >= 2.0 else 'ja/unklar'}
- Disk-Bottleneck ja/nein: nein, vorher gemessen 2.8 GB/s write und 3.7 GB/s read

## 7. Nächster Schritt
- echtes Boot-Profil image-only oder video-only mit Manifest-Speed messen
"""
    MARKDOWN_REPORT.write_text(md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a controlled 1-3 GB HF/Xet download benchmark.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--metadata-only", action="store_true", help="Validate metadata and write no benchmark run.")
    args = parser.parse_args()

    preflight_status = preflight()
    metadata = get_file_metadata(args.repo_id, args.filename)
    print(f"repo_id={args.repo_id}")
    print(f"filename={args.filename}")
    print(f"bytes={metadata['bytes']}")
    print(f"gated={metadata.get('gated')} private={metadata.get('private')}")
    if args.metadata_only:
        return 0

    existing = load_existing_report()
    run = run_download(args.repo_id, args.filename, metadata, preflight_status)
    payload = {
        "created_at": existing.get("created_at") or utc_now(),
        "updated_at": utc_now(),
        "environment": preflight_status,
        "testfile": metadata,
        "runs": existing.get("runs", []) + [run],
        "cleanup_note": "Benchmark file is in the HF cache and may be deleted later if cache-hit validation is no longer needed.",
    }
    write_reports(payload)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
