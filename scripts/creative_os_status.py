#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_core.creative_os.dashboard import render_dashboard, render_rich_dashboard
from agent_core.creative_os.run_inspector import CreativeOSRunInspector


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Creative OS run dashboard.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--view",
        choices=("overview", "skills", "stages", "artifacts", "issues", "next", "all"),
        default="overview",
    )
    parser.add_argument(
        "--focus",
        choices=("cli", "render", "audit", "none"),
        default="none",
        help="Operator focus label for the read-only dashboard.",
    )
    parser.add_argument(
        "--style",
        choices=("plain", "rich"),
        default="plain",
        help="Dashboard rendering style. Rich falls back to plain if the package is unavailable.",
    )
    args = parser.parse_args()
    inspector = CreativeOSRunInspector()
    inspection = inspector.inspect(args.job_id)
    if args.style == "rich":
        print(render_rich_dashboard(inspection, view=args.view, focus=args.focus), end="")
    else:
        print(render_dashboard(inspection, view=args.view, focus=args.focus), end="")
    if not inspection.exists:
        return 1
    if inspection.blocking_issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
