from __future__ import annotations

import json
from unittest import mock
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_core.creative_os.dashboard import render_dashboard, render_rich_dashboard
from agent_core.creative_os.run_inspector import CreativeOSRunInspector


SOURCE_RUN = Path("/workspace/agent_runs/creative-os-jungle-001/creative_os")


class CreativeOSStatusTests(unittest.TestCase):
    def test_inspector_detects_existing_artifacts_and_next_action(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        self.assertTrue(inspection.exists)
        self.assertEqual("ready_for_ltx_i2v_takes", inspection.status)
        self.assertFalse(inspection.blocking_issues)
        self.assertTrue(inspection.artifacts["ltx_motion_prompts.json"])
        self.assertIn("Stage 09", CreativeOSRunInspector().next_action(inspection))

    def test_missing_files_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "partial" / "creative_os"
            run_dir.mkdir(parents=True)
            (run_dir / "normalized_job.json").write_text("{}", encoding="utf-8")
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("partial")
            self.assertTrue(inspection.exists)
            self.assertEqual("passed", inspection.stages[0].status)
            self.assertEqual("missing", inspection.stages[1].status)
            self.assertIn("Input normalized", render_dashboard(inspection, view="overview"))
            self.assertIn("CURRENT POSITION", render_dashboard(inspection, view="overview"))

    def test_needs_review_or_rejected_generates_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "issue-run" / "creative_os"
            shutil.copytree(SOURCE_RUN, run_dir)
            reviews = json.loads((run_dir / "keyframe_review.json").read_text(encoding="utf-8"))
            reviews[0]["status"] = "rejected"
            reviews[0]["issues"] = ["visible text"]
            (run_dir / "keyframe_review.json").write_text(json.dumps(reviews), encoding="utf-8")
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("issue-run")
            self.assertTrue(inspection.blocking_issues)
            self.assertEqual("blocked_by_keyframe_review", inspection.status)
            self.assertIn("visible text", render_dashboard(inspection, view="issues"))

    def test_skill_health_separates_optional_and_blocking_missing(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        health = CreativeOSRunInspector().skill_health(inspection)
        self.assertEqual(["modes/jungle_adventure"], health["missing_optional"])
        self.assertEqual([], health["blocking_missing"])
        self.assertEqual("ok", health["status"])
        dashboard = render_dashboard(inspection, view="skills")
        self.assertIn("missing optional:", dashboard)
        self.assertIn("blocking missing:", dashboard)

    def test_overview_uses_cockpit_layout(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        dashboard = render_dashboard(inspection, view="overview", focus="cli")
        self.assertIn("╭", dashboard)
        self.assertIn("│ CONTENT MASCHINE LIVE", dashboard)
        self.assertIn("CURRENT POSITION", dashboard)
        self.assertIn("✓ 08 LTX motion prompts", dashboard)
        self.assertIn("○ 09 LTX video takes", dashboard)
        self.assertIn("→ Z-Image Prompts", dashboard)
        self.assertIn("SKILL HEALTH", dashboard)
        self.assertIn("✓ ok · loaded 8 · fallbacks 2 · missing optional 1 · blocking 0", dashboard)
        self.assertIn("✓ 3 keyframes", dashboard)
        self.assertNotIn("keyframes/scene_01.png", dashboard)

    def test_all_view_uses_clear_separators(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        dashboard = render_dashboard(inspection, view="all", focus="cli")
        self.assertIn("────────────────", dashboard)
        self.assertIn("SKILLS\n────────────────", dashboard)
        self.assertIn("ARTIFACTS\n────────────────", dashboard)

    def test_read_only_inspection_does_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "readonly" / "creative_os"
            shutil.copytree(SOURCE_RUN, run_dir)
            before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("readonly")
            render_dashboard(inspection, view="all", focus="cli")
            after = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())
            self.assertEqual(before, after)

    def test_cli_overview_and_all_execute(self) -> None:
        overview = subprocess.run(
            [
                "python3",
                "/workspace/scripts/creative_os_status.py",
                "--job-id",
                "creative-os-jungle-001",
                "--view",
                "overview",
                "--focus",
                "cli",
                "--style",
                "plain",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, overview.returncode, overview.stderr)
        self.assertIn("CONTENT MASCHINE LIVE", overview.stdout)
        self.assertIn("╭", overview.stdout)
        self.assertIn("Status     ready_for_ltx_i2v_takes", overview.stdout)
        self.assertIn("Render paused    yes", overview.stdout)
        self.assertIn("Improve CLI cockpit before rendering", overview.stdout)
        self.assertIn("Video Backend    ? not_checked · planned ltx2", overview.stdout)
        self.assertIn("SKILL HEALTH", overview.stdout)
        self.assertIn("✓ 3 keyframes", overview.stdout)
        all_view = subprocess.run(
            [
                "python3",
                "/workspace/scripts/creative_os_status.py",
                "--job-id",
                "creative-os-jungle-001",
                "--view",
                "all",
                "--focus",
                "cli",
                "--style",
                "plain",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, all_view.returncode, all_view.stderr)
        self.assertIn("ARTIFACTS", all_view.stdout)
        self.assertIn("NEXT", all_view.stdout)
        self.assertIn("────────────────", all_view.stdout)

    def test_rich_overview_contains_cockpit_panels(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        dashboard = render_rich_dashboard(inspection, view="overview", focus="cli")
        self.assertIn("CONTENT MASCHINE LIVE", dashboard)
        self.assertIn("SYSTEM STATUS", dashboard)
        self.assertIn("PIPELINE MAP", dashboard)
        self.assertIn("ACTIVE WORKSPACE", dashboard)
        self.assertIn("SCENE JOBS", dashboard)
        self.assertIn("LTX Motion Ready", dashboard)
        self.assertIn("scene_01", dashboard)
        self.assertIn("READY FOR LTX", dashboard)
        self.assertIn("Stage output:", dashboard)
        self.assertIn("SKILL HEALTH", dashboard)
        self.assertIn("ARTIFACTS", dashboard)
        self.assertIn("ISSUES", dashboard)
        self.assertIn("NEXT", dashboard)
        self.assertNotIn("%", dashboard)
        self.assertNotIn("ETA", dashboard)

    def test_rich_missing_package_falls_back_to_plain(self) -> None:
        inspection = CreativeOSRunInspector().inspect("creative-os-jungle-001")
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            dashboard = render_rich_dashboard(inspection, view="overview", focus="cli")
        self.assertIn("Rich is not installed; falling back to plain dashboard.", dashboard)
        self.assertIn("CONTENT MASCHINE LIVE", dashboard)

    def test_rich_style_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "readonly-rich" / "creative_os"
            shutil.copytree(SOURCE_RUN, run_dir)
            before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("readonly-rich")
            render_rich_dashboard(inspection, view="overview", focus="cli")
            after = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())
            self.assertEqual(before, after)

    def test_cli_rich_overview_and_all_execute(self) -> None:
        overview = subprocess.run(
            [
                "python3",
                "/workspace/scripts/creative_os_status.py",
                "--job-id",
                "creative-os-jungle-001",
                "--view",
                "overview",
                "--focus",
                "cli",
                "--style",
                "rich",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, overview.returncode, overview.stderr)
        self.assertIn("ACTIVE WORKSPACE", overview.stdout)
        self.assertIn("SCENE JOBS", overview.stdout)
        all_view = subprocess.run(
            [
                "python3",
                "/workspace/scripts/creative_os_status.py",
                "--job-id",
                "creative-os-jungle-001",
                "--view",
                "all",
                "--focus",
                "cli",
                "--style",
                "rich",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, all_view.returncode, all_view.stderr)
        self.assertIn("SYSTEM STATUS", all_view.stdout)
        self.assertIn("SKILLS", all_view.stdout)


if __name__ == "__main__":
    unittest.main()
