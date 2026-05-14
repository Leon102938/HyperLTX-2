from __future__ import annotations

import json
from dataclasses import replace
from unittest import mock
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import textual

from agent_core.creative_os.dashboard import render_dashboard, render_rich_dashboard
from agent_core.creative_os.run_inspector import CreativeOSRunInspector
from agent_core.creative_os.cockpit.state_adapter import CockpitStateAdapter
from agent_core.creative_os.cockpit.stage_registry import current_stage_id
from agent_core.creative_os.cockpit.panels import active_workspace_panel, pipeline_map_panel


FIXTURE_RUNS_ROOT = Path("/workspace/tests/fixtures/creative_os_runs")
SOURCE_RUN = FIXTURE_RUNS_ROOT / "creative-os-jungle-001" / "creative_os"


class CreativeOSStatusTests(unittest.TestCase):
    def test_textual_version_stays_on_089(self) -> None:
        self.assertTrue(textual.__version__.startswith("0.89."), textual.__version__)

    def test_phase1_cli_creates_stage00_to_stage09_artifacts_without_fake_backend_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "run-phase1",
                    "--job-id",
                    "phase1-test",
                    "--topic",
                    "jungle safari at sunrise",
                    "--pipeline",
                    "shortform_storyboard_v1",
                    "--mode",
                    "visual_adventure",
                    "--style",
                    "cinematic_nature",
                    "--format",
                    "portrait",
                    "--duration",
                    "9s",
                    "--scenes",
                    "3",
                    "--runs-root",
                    tmp,
                    "--no-images",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            run_dir = Path(tmp) / "phase1-test" / "creative_os"
            for artifact in (
                "normalized_job.json",
                "pipeline_route.json",
                "mode_style.json",
                "skill_match.json",
                "skill_tree.json",
                "creative_strategy.json",
                "beat_hook_plan.json",
                "creative_judge.json",
                "scene_contracts.json",
                "prompt_payload_compiled.json",
                "zimage_prompts.json",
                "keyframe_manifest.json",
                "phase1_status.json",
            ):
                self.assertTrue((run_dir / artifact).exists(), artifact)

            manifest = json.loads((run_dir / "keyframe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("missing", manifest["backend_status"])
            self.assertEqual(3, len(manifest["jobs"]))
            self.assertTrue(all(job["status"] == "error" for job in manifest["jobs"]))
            self.assertFalse(any(Path(job["output_path"]).exists() for job in manifest["jobs"]))

            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("phase1-test")
            self.assertEqual("phase1_paused_missing_image_backend", inspection.status)
            state = CockpitStateAdapter(job_id="phase1-test", runs_root=tmp).load()
            self.assertEqual("creative_os", state.run_type)
            self.assertEqual("jungle safari at sunrise", state.header.topic)
            self.assertEqual("visual_adventure", state.header.mode)
            self.assertEqual("error", state.workspace.scenes[0].status)
            self.assertEqual("09", current_stage_id(state))

    def test_phase1_live_cli_writes_live_status_and_events_without_fake_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "run-phase1-live",
                    "--job-id",
                    "phase1-live-test",
                    "--topic",
                    "jungle safari at sunrise",
                    "--pipeline",
                    "shortform_storyboard_v1",
                    "--mode",
                    "visual_adventure",
                    "--style",
                    "cinematic_nature",
                    "--format",
                    "portrait",
                    "--duration",
                    "9s",
                    "--scenes",
                    "3",
                    "--runs-root",
                    tmp,
                    "--no-images",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            run_dir = Path(tmp) / "phase1-live-test" / "creative_os"
            live = json.loads((run_dir / "live_status.json").read_text(encoding="utf-8"))
            self.assertEqual("00", live["viewed_stage"])
            self.assertEqual("09", live["real_run_stage"])
            self.assertIsNone(live["current_running_stage"])
            self.assertEqual([f"{index:02d}" for index in range(9)], live["completed_stages"])
            self.assertEqual(["09"], live["failed_stages"])
            self.assertEqual([], live["pending_stages"])
            self.assertEqual("done", live["stages"]["00"]["status"])
            self.assertEqual("error", live["stages"]["09"]["status"])
            self.assertEqual("paused_missing_backend", live["status"])
            self.assertTrue((run_dir / "stage_events.jsonl").exists())
            events = [json.loads(line) for line in (run_dir / "stage_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(("init", "00", "pending"), (events[0]["kind"], events[0]["stage"], events[0]["status"]))
            self.assertIn(("09", "running"), [(event["stage"], event["status"]) for event in events])
            self.assertIn(("09", "error"), [(event["stage"], event["status"]) for event in events])
            self.assertNotIn(("09", "done"), [(event["stage"], event["status"]) for event in events])

            phase1 = json.loads((run_dir / "phase1_status.json").read_text(encoding="utf-8"))
            self.assertEqual("paused_missing_backend", phase1["status"])
            self.assertEqual([f"{index:02d}" for index in range(9)], phase1["completed_stages"])
            self.assertEqual("08", phase1["last_completed_stage"])
            self.assertEqual("09", phase1["current_stage"])
            self.assertEqual("09", phase1["real_run_stage"])
            self.assertEqual("09", phase1["next_available_stage"])
            self.assertEqual("not_built", phase1["stage10_plus"])

            manifest = json.loads((run_dir / "keyframe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("missing", manifest["backend_status"])
            self.assertTrue(all(job["status"] == "error" for job in manifest["jobs"]))
            self.assertTrue(all(job["progress_percent"] is None for job in manifest["jobs"]))

            state = CockpitStateAdapter(job_id="phase1-live-test", runs_root=tmp, watch_enabled=True, refresh_sec=1).load()
            self.assertEqual("00", state.selected_stage)
            self.assertIn("Viewed 00 / Real Stage 09", state.workspace.current_step)
            self.assertEqual("Stage 09 paused / image backend unavailable", state.workspace.next_technical)
            self.assertEqual("", state.workspace.scenes[0].progress_percent)
            self.assertEqual("", state.workspace.scenes[0].output_path)
            pipeline = str(pipeline_map_panel.render(state))
            self.assertIn("✗ 09 Image / Keyframe Generation", pipeline)
            self.assertNotIn("✓ 09 Image / Keyframe Generation", pipeline)
            self.assertNotIn("✓ 10 Keyframe Review", pipeline)
            workspace = str(active_workspace_panel.render(replace(state, selected_stage="09")))
            self.assertIn("KEYFRAME MANIFEST / LIVE STATE", workspace)
            self.assertIn("Backend Status", workspace)
            self.assertIn("missing", workspace)
            self.assertIn("Backend Reason", workspace)
            self.assertIn("disabled_by_cli", workspace)

    def test_phase1_live_accepts_stage_delay_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "run-phase1-live",
                    "--job-id",
                    "phase1-live-delay",
                    "--topic",
                    "jungle safari at sunrise",
                    "--runs-root",
                    tmp,
                    "--no-generate-images",
                    "--stage-delay-seconds",
                    "0",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue((Path(tmp) / "phase1-live-delay" / "creative_os" / "live_status.json").exists())

    def test_open_cockpit_without_tty_does_not_start_textual_or_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "run-phase1-live",
                    "--job-id",
                    "phase1-open-cockpit-no-tty",
                    "--topic",
                    "jungle safari at sunrise",
                    "--runs-root",
                    tmp,
                    "--open-cockpit",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode)
            self.assertNotIn("OSError", result.stderr)
            self.assertIn("--open-cockpit is disabled", result.stderr)
            self.assertIn("Terminal 1:", result.stderr)
            self.assertIn("Terminal 2:", result.stderr)
            self.assertFalse((Path(tmp) / "phase1-open-cockpit-no-tty").exists())

    def test_live_stage_missing_is_missing_not_fake_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_live_no_images(tmp, "phase1-live-missing")
            run_dir = Path(tmp) / "phase1-live-missing" / "creative_os"
            (run_dir / "skill_tree.json").unlink()
            live = json.loads((run_dir / "live_status.json").read_text(encoding="utf-8"))
            live["stages"]["03"]["status"] = "missing"
            live["stages"]["03"]["error"] = "missing artifact: skill_tree.json"
            live["completed_stages"].remove("03")
            live["missing_stages"] = ["03"]
            (run_dir / "live_status.json").write_text(json.dumps(live), encoding="utf-8")

            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("phase1-live-missing")
            stage03 = next(stage for stage in inspection.stages if stage.index == "03")
            self.assertEqual("missing", stage03.status)
            self.assertIn("skill_tree.json", stage03.detail)
            state = CockpitStateAdapter(job_id="phase1-live-missing", runs_root=tmp).load()
            self.assertEqual("missing", [stage.status for stage in inspection.stages if stage.index == "03"][0])
            self.assertEqual("03", state.workspace.next_technical.split()[1])

    def test_phase1_finished_status_completes_stage09_and_stage09_workspace_uses_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-finished")
            run_dir = Path(tmp) / "phase1-finished" / "creative_os"
            manifest = json.loads((run_dir / "keyframe_manifest.json").read_text(encoding="utf-8"))
            for job in manifest["jobs"]:
                output_path = Path(job["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake-png-test")
                job.update(
                    {
                        "status": "finished",
                        "progress_percent": 100,
                        "error": None,
                        "backend_job_id": f"test_{job['scene_id']}",
                        "file_exists": True,
                        "file_size_bytes": output_path.stat().st_size,
                        "file_mtime": "2026-05-13T00:00:00Z",
                    }
                )
            manifest["backend_status"] = "available"
            manifest["backend_reason"] = "ready"
            manifest["overall_status"] = "finished"
            manifest["gallery_path"] = str(run_dir / "keyframe_gallery.html")
            (run_dir / "keyframe_gallery.html").write_text("<html>gallery</html>\n", encoding="utf-8")
            (run_dir / "keyframe_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            phase1_status = json.loads((run_dir / "phase1_status.json").read_text(encoding="utf-8"))
            phase1_status.update(
                {
                    "status": "finished",
                    "current_stage": "09",
                    "real_run_stage": "09",
                    "last_completed_stage": "09",
                    "next_available_stage": "none_phase1_complete",
                    "completed_stages": [f"{index:02d}" for index in range(10)],
                    "stage10_plus": "not_built",
                }
            )
            (run_dir / "phase1_status.json").write_text(json.dumps(phase1_status), encoding="utf-8")

            self.assertIn("09", phase1_status["completed_stages"])
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("phase1-finished")
            self.assertEqual("phase1_finished_stage09", inspection.status)
            state = CockpitStateAdapter(job_id="phase1-finished", runs_root=tmp).load()
            self.assertEqual("09", current_stage_id(state))
            self.assertEqual("Phase 1 complete / Stage 10+ not built yet", state.workspace.next_technical)
            workspace = str(active_workspace_panel.render(state))
            pipeline = str(pipeline_map_panel.render(state))
            self.assertIn("Preview: keyframes/scene_01.png", workspace)
            self.assertIn("Preview: keyframes/scene_02.png", workspace)
            self.assertIn("Preview: keyframes/scene_03.png", workspace)
            self.assertIn("KEYFRAME MANIFEST / LIVE STATE", workspace)
            self.assertIn("Overall Status", workspace)
            self.assertIn("finished", workspace)
            self.assertIn("path ok", workspace)
            self.assertIn("Gallery: keyframe_gallery.html", workspace)
            self.assertNotIn("✓ 10 Keyframe Review", pipeline)

    def test_finished_stage09_job_missing_output_is_error_not_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-missing-output")
            run_dir = Path(tmp) / "phase1-missing-output" / "creative_os"
            manifest = json.loads((run_dir / "keyframe_manifest.json").read_text(encoding="utf-8"))
            manifest["backend_status"] = "available"
            manifest["overall_status"] = "finished"
            manifest["jobs"][0]["status"] = "finished"
            manifest["jobs"][0]["error"] = None
            missing_path = Path(manifest["jobs"][0]["output_path"])
            if missing_path.exists():
                missing_path.unlink()
            (run_dir / "keyframe_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            inspection = CreativeOSRunInspector(runs_root=tmp).inspect("phase1-missing-output")
            stage = next(stage for stage in inspection.stages if stage.artifact == "keyframe_manifest.json")
            self.assertEqual("needs_review", stage.status)
            state = CockpitStateAdapter(job_id="phase1-missing-output", runs_root=tmp).load()
            self.assertEqual("error", state.workspace.scenes[0].status)
            pipeline = str(pipeline_map_panel.render(state))
            self.assertNotIn("✓ 09 Image / Keyframe Generation", pipeline)
            workspace = str(active_workspace_panel.render(state))
            self.assertIn("path missing", workspace)
            self.assertIn("finished job output missing", workspace)

    def test_retry_keyframes_dry_run_detects_failed_jobs_and_scene_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-retry-dry-run")
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "retry-keyframes",
                    "--job-id",
                    "phase1-retry-dry-run",
                    "--runs-root",
                    tmp,
                    "--dry-run",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("dry_run", payload["status"])
            self.assertEqual(3, len(payload["retry_jobs"]))
            self.assertEqual({"scene_01", "scene_02", "scene_03"}, {item["scene_id"] for item in payload["retry_jobs"]})

            scene_result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "retry-keyframes",
                    "--job-id",
                    "phase1-retry-dry-run",
                    "--runs-root",
                    tmp,
                    "--scene",
                    "scene_02",
                    "--dry-run",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, scene_result.returncode, scene_result.stderr)
            scene_payload = json.loads(scene_result.stdout)
            self.assertEqual(["scene_02"], [item["scene_id"] for item in scene_payload["retry_jobs"]])

    def test_retry_keyframes_does_not_rewrite_stage00_to_stage08(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-retry-no-rewrite")
            run_dir = Path(tmp) / "phase1-retry-no-rewrite" / "creative_os"
            protected = [
                run_dir / "normalized_job.json",
                run_dir / "pipeline_route.json",
                run_dir / "mode_style.json",
                run_dir / "skill_tree.json",
                run_dir / "creative_strategy.json",
                run_dir / "beat_hook_plan.json",
                run_dir / "creative_judge.json",
                run_dir / "scene_contracts.json",
                run_dir / "prompt_payload_compiled.json",
            ]
            before = {path.name: path.stat().st_mtime_ns for path in protected}
            result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "retry-keyframes",
                    "--job-id",
                    "phase1-retry-no-rewrite",
                    "--runs-root",
                    tmp,
                    "--image-backend-url",
                    "http://127.0.0.1:1",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            after = {path.name: path.stat().st_mtime_ns for path in protected}
            self.assertEqual(before, after)
            payload = json.loads(result.stdout)
            self.assertEqual(["keyframe_manifest.json", "phase1_status.json"], payload["updated_files"])

    def test_retry_keyframes_protects_finished_outputs_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-retry-force")
            run_dir = Path(tmp) / "phase1-retry-force" / "creative_os"
            manifest = json.loads((run_dir / "keyframe_manifest.json").read_text(encoding="utf-8"))
            for job in manifest["jobs"]:
                output_path = Path(job["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake-png-test")
                job.update({"status": "finished", "progress_percent": 100, "error": None, "file_exists": True, "file_size_bytes": output_path.stat().st_size})
            manifest["backend_status"] = "available"
            manifest["overall_status"] = "finished"
            (run_dir / "keyframe_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            protected_result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "retry-keyframes",
                    "--job-id",
                    "phase1-retry-force",
                    "--runs-root",
                    tmp,
                    "--dry-run",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, protected_result.returncode, protected_result.stderr)
            self.assertEqual([], json.loads(protected_result.stdout)["retry_jobs"])

            forced_result = subprocess.run(
                [
                    "python3",
                    "/workspace/scripts/agent_core_cli.py",
                    "creative-os",
                    "retry-keyframes",
                    "--job-id",
                    "phase1-retry-force",
                    "--runs-root",
                    tmp,
                    "--scene",
                    "scene_02",
                    "--force",
                    "--dry-run",
                    "--print-json",
                ],
                cwd="/workspace",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, forced_result.returncode, forced_result.stderr)
            forced = json.loads(forced_result.stdout)
            self.assertEqual([("scene_02", "force")], [(item["scene_id"], item["reason"]) for item in forced["retry_jobs"]])

    def test_missing_stage09_manifest_does_not_render_fake_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_phase1_no_images(tmp, "phase1-missing-manifest")
            run_dir = Path(tmp) / "phase1-missing-manifest" / "creative_os"
            (run_dir / "keyframe_manifest.json").unlink()
            state = CockpitStateAdapter(job_id="phase1-missing-manifest", runs_root=tmp).load()
            workspace = str(active_workspace_panel.render(state))
            self.assertIn("missing manifest: keyframe_manifest.json unavailable", workspace)
            self.assertNotIn("Image 1 / scene_01", workspace)

    def _run_phase1_no_images(self, runs_root: str, job_id: str) -> None:
        result = subprocess.run(
            [
                "python3",
                "/workspace/scripts/agent_core_cli.py",
                "creative-os",
                "run-phase1",
                "--job-id",
                job_id,
                "--topic",
                "jungle safari at sunrise",
                "--runs-root",
                runs_root,
                "--no-images",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def _run_phase1_live_no_images(self, runs_root: str, job_id: str) -> None:
        result = subprocess.run(
            [
                "python3",
                "/workspace/scripts/agent_core_cli.py",
                "creative-os",
                "run-phase1-live",
                "--job-id",
                job_id,
                "--topic",
                "jungle safari at sunrise",
                "--runs-root",
                runs_root,
                "--no-images",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_inspector_detects_existing_artifacts_and_next_action(self) -> None:
        inspector = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT)
        inspection = inspector.inspect("creative-os-jungle-001")
        self.assertTrue(inspection.exists)
        self.assertEqual("ready_for_ltx_i2v_takes", inspection.status)
        self.assertFalse(inspection.blocking_issues)
        self.assertTrue(inspection.artifacts["ltx_motion_prompts.json"])
        self.assertIn("Stage 09", inspector.next_action(inspection))

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

    def test_missing_run_message_explains_disposable_agent_runs(self) -> None:
        inspection = CreativeOSRunInspector(runs_root="/workspace/agent_runs").inspect("definitely-missing-run")
        dashboard = render_rich_dashboard(inspection, view="overview")
        self.assertIn("Run not found:", dashboard)
        self.assertIn("job_id: definitely-missing-run", dashboard)
        self.assertIn("searched: /workspace/agent_runs/definitely-missing-run/creative_os", dashboard)
        self.assertIn("This is not a system error.", dashboard)
        self.assertIn("agent_runs contains disposable run artifacts only.", dashboard)
        self.assertIn("Use --runs-root for fixtures or create a real run first.", dashboard)

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
        inspector = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT)
        inspection = inspector.inspect("creative-os-jungle-001")
        health = inspector.skill_health(inspection)
        self.assertEqual(["modes/jungle_adventure"], health["missing_optional"])
        self.assertEqual([], health["blocking_missing"])
        self.assertEqual("ok", health["status"])
        dashboard = render_dashboard(inspection, view="skills")
        self.assertIn("missing optional:", dashboard)
        self.assertIn("blocking missing:", dashboard)

    def test_overview_uses_cockpit_layout(self) -> None:
        inspection = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT).inspect("creative-os-jungle-001")
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
        inspection = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT).inspect("creative-os-jungle-001")
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
                "--runs-root",
                str(FIXTURE_RUNS_ROOT),
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
                "--runs-root",
                str(FIXTURE_RUNS_ROOT),
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
        inspection = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT).inspect("creative-os-jungle-001")
        dashboard = render_rich_dashboard(inspection, view="overview", focus="cli")
        self.assertIn("CONTENT MASCHINE LIVE", dashboard)
        self.assertIn("fixture/demo", dashboard)
        self.assertIn("SYSTEM STATUS", dashboard)
        self.assertIn("PIPELINE MAP", dashboard)
        self.assertIn("ACTIVE WORKSPACE", dashboard)
        self.assertIn("SCENE JOBS", dashboard)
        self.assertIn("LTX Motion Ready", dashboard)
        self.assertIn("scene_01", dashboard)
        self.assertIn("READY FOR LTX", dashboard)
        self.assertIn("Stage output:", dashboard)
        self.assertIn("TECHNICAL FLOW", dashboard)
        self.assertIn("SKILL HEALTH", dashboard)
        self.assertIn("ARTIFACTS", dashboard)
        self.assertIn("ISSUES", dashboard)
        self.assertIn("NEXT", dashboard)
        self.assertNotIn("%", dashboard)
        self.assertNotIn("ETA", dashboard)

    def test_rich_missing_package_falls_back_to_plain(self) -> None:
        inspection = CreativeOSRunInspector(runs_root=FIXTURE_RUNS_ROOT).inspect("creative-os-jungle-001")
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
                "--runs-root",
                str(FIXTURE_RUNS_ROOT),
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
                "--runs-root",
                str(FIXTURE_RUNS_ROOT),
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
