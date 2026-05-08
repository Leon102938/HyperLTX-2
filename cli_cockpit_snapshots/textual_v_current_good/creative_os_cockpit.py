#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Textual Creative OS cockpit prototype.")
    parser.add_argument("--job-id")
    parser.add_argument(
        "--theme-preview",
        action="store_true",
        help="Open a static Textual theme preview without reading run data.",
    )
    parser.add_argument(
        "--runs-root",
        default="/workspace/agent_runs",
        help="Root containing disposable Creative OS run artifacts. Use /workspace/tests/fixtures/creative_os_runs for design checks.",
    )
    args = parser.parse_args()

    os.environ.setdefault("COLORTERM", "truecolor")

    try:
        from agent_core.creative_os.textual_cockpit import run_cockpit, run_theme_preview
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.theme_preview:
        run_theme_preview()
        return 0

    if not args.job_id:
        parser.error("--job-id is required unless --theme-preview is used")

    run_cockpit(args.job_id, args.runs_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
