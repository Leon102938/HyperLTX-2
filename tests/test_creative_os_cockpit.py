from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_core.creative_os.phase1_runtime import Phase1RunConfig, run_phase1_live
from agent_core.creative_os.textual_cockpit import CockpitArgs, CreativeOSCockpitApp, ThemePreviewApp, _scene_card_text
from agent_core.creative_os.cockpit.panel_registry import PANEL_CONFIG, PANEL_REGISTRY, enabled_panels
from agent_core.creative_os.cockpit.stage_registry import STAGE_DEFINITIONS, stage_ids
from agent_core.creative_os.cockpit.state_adapter import CockpitStateAdapter
from agent_core.creative_os.cockpit.panels import (
    active_workspace_panel,
    artifacts_panel,
    header_panel,
    issues_panel,
    next_panel,
    pipeline_map_panel,
    skill_health_panel,
    system_status_panel,
)


FIXTURE_RUNS_ROOT = Path("/workspace/tests/fixtures/creative_os_runs")


class CreativeOSCockpitTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_cockpit_loads_fixture_layout(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test() as pilot:
            self.assertEqual("ready_for_ltx_i2v_takes", app.inspection.status)
            self.assertEqual("fixture/demo", app.state.session_mode)
            self.assertEqual("creative_os", app.state.run_type)
            self.assertEqual(FIXTURE_RUNS_ROOT / "creative-os-jungle-001", app.state.data_source_path)
            self.assertTrue(app.state.run_found)
            self.assertFalse(app.state.watch_enabled)
            text = _widget_text(app.query_one("#workspace"))
            self.assertIn("CURRENT POSITION", str(text))
            self.assertNotIn("PIPELINE PATH", str(text))
            self.assertIn("PROMPTS / IMAGE JOBS", str(text))
            self.assertNotIn("PIPELINE FLOW", str(text))
            self.assertNotIn("RUN NOTES", str(text))
            self.assertIn("FINAL MP4", str(text))
            self.assertIn("DIRECTOR MODE", str(text))
            self.assertIn("CURRENT STEP", str(text))
            self.assertIn("OPERATOR FOCUS", str(text))
            self.assertIn("RENDER PAUSED", str(text))
            self.assertIn("LAST PASSED", str(text))
            self.assertIn("NEXT TECHNICAL", str(text))
            self.assertIn("RUN TYPE", str(text))
            self.assertIn("not_checked", str(text))
            self.assertIn("scene_01", str(text))
            self.assertIn("scene_03", str(text))
            self.assertIn("ready", str(text))
            self.assertIn("Image 1 / scene_01", str(text))
            self.assertIn("Image 2 / scene_02", str(text))
            self.assertIn("Image 3 / scene_03", str(text))
            self.assertIn("╭", str(text))
            self.assertIn("╰", str(text))
            self.assertNotIn("IMAGE JOB 01", str(text))
            self.assertNotIn("JOB / PROMPT", str(text))
            self.assertNotIn("details  ", str(text))
            self.assertIn("Prompt:", str(text))
            self.assertIn("Preview:", str(text))
            self.assertIn(">", str(text))
            self.assertIn("v", str(text))
            self.assertNotIn("negative_prompt", str(text))
            brand = _widget_text(app.query_one("#header-brand"))
            self.assertIn("CONTENT MASCHINE LIVE", brand)
            self.assertNotIn("▛", brand)
            header_details = _widget_text(app.query_one("#header-details"))
            self.assertIn("JOB        creative-os-jungle-001", header_details)
            self.assertIn("PIPELINE   shortform_storyboard_v1", header_details)
            self.assertNotIn("JOBcreative-os-jungle-001", header_details)
            self.assertNotIn("PIPELINEshortform_storyboard_v1", header_details)
            self.assertIn("fixture/demo", _widget_text(app.query_one("#header-meta")))
            self.assertIn("Watch off", _widget_text(app.query_one("#keybar")))
            await pilot.press("h")
            self.assertTrue(app.query_one("#help-panel").has_class("visible"))
            await pilot.press("r")
            self.assertEqual("ready_for_ltx_i2v_takes", app.inspection.status)
            self.assertNotIn("/workspace/agent_runs", str(app.inspection.run_dir))

    def test_stage_registry_contains_pipeline_map_v1(self) -> None:
        self.assertEqual(tuple(f"{index:02d}" for index in range(16)), stage_ids())
        self.assertEqual("Command Center", STAGE_DEFINITIONS[0].title)
        self.assertEqual("Pipeline wählen", STAGE_DEFINITIONS[1].title)
        self.assertEqual("Mode & Style", STAGE_DEFINITIONS[2].title)
        self.assertEqual("Skills laden", STAGE_DEFINITIONS[3].title)
        self.assertEqual("Image / Keyframe Generation", STAGE_DEFINITIONS[9].title)
        self.assertEqual("Final Output", STAGE_DEFINITIONS[15].title)

    async def test_pipeline_map_v1_renders_all_stages_and_selection(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test() as pilot:
            pipeline = _widget_text(app.query_one("#pipeline-map"))
            for stage in STAGE_DEFINITIONS:
                self.assertIn(f"{stage.stage_id} {stage.title}", pipeline)
            self.assertEqual("09", app.state.selected_stage)
            self.assertIn("▸ 09 Image / Keyframe Generation", pipeline)
            await pilot.press("down")
            self.assertEqual("10", app.state.selected_stage)
            self.assertIn("▸ 10 Keyframe Review", _widget_text(app.query_one("#pipeline-map")))
            await pilot.press("k")
            self.assertEqual("09", app.state.selected_stage)

    async def test_watch_refresh_does_not_override_manual_stage_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            run_phase1_live(
                Phase1RunConfig(
                    job_id="phase1-watch-selection",
                    topic="jungle safari at sunrise",
                    runs_root=runs_root,
                    attempt_images=False,
                )
            )
            app = CreativeOSCockpitApp(CockpitArgs(job_id="phase1-watch-selection", runs_root=runs_root, watch=True, refresh_sec=1))
            async with app.run_test():
                self.assertEqual("00", app.state.selected_stage)
                app._select_stage("05")
                self.assertEqual("05", app.state.selected_stage)
                live_path = runs_root / "phase1-watch-selection" / "creative_os" / "live_status.json"
                live = json.loads(live_path.read_text(encoding="utf-8"))
                live["updated_at"] = "2026-05-14T12:00:00Z"
                live_path.write_text(json.dumps(live), encoding="utf-8")
                app._watch_refresh()
                self.assertEqual("05", app.state.selected_stage)

    async def test_active_workspace_stage_router_views(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test():
            expected = {
                "00": ("COMMAND CENTER", "COMMAND COMPOSER", "Run planned / disabled in V0.2", "COMMAND PREVIEW"),
                "01": ("PIPELINE WÄHLEN", "PIPELINE PURPOSE / OVERVIEW", "PIPELINE FLOW", "PIPELINE ASSETS"),
                "02": ("MODE & STYLE", "visual_adventure", "MODE INTENT / STYLE LANGUAGE / HANDOFF"),
                "03": ("SKILLS LADEN", "SKILL TREE V1", "SKILL LOADING PROGRESS"),
            }
            for stage_id, needles in expected.items():
                app._select_stage(stage_id)
                workspace = _widget_text(app.query_one("#workspace"))
                for needle in needles:
                    self.assertIn(needle, workspace)
                if stage_id == "00":
                    self.assertIn("Run vorbereiten · Parameter prüfen · Startstatus kontrollieren", workspace)
                    self.assertIn("DEV / RUN INFO", workspace)
                    self.assertIn("Run State: ready_for_ltx_i2v_takes", workspace)
                    self.assertNotIn("DATA SOURCE", workspace)
                    self.assertNotIn("Status: ready_for_ltx_i2v_takes", workspace)
                    self.assertNotIn("CLI-Befehl spaeter bauen", workspace)
                if stage_id == "01":
                    self.assertNotIn("CURRENT POSITION", workspace)
                    self.assertNotIn("OUTPUT TARGETS / NEXT", workspace)
                    self.assertNotIn("Output Goals", workspace)
                    self.assertNotIn("Next Step", workspace)
                    self.assertNotIn("Operator Next", workspace)
                    self.assertGreater(workspace.find("PIPELINE ASSETS"), workspace.find("PIPELINE PURPOSE / OVERVIEW"))
                    self.assertIn("01 Pipeline overview        selected pipeline route          active", workspace)
                    self.assertIn("STATUS KEY  active = selected  done = passed  upcoming = next  locked = later", workspace)
                    self.assertIn("direction inputs                 upcoming", workspace)
                    self.assertIn("Video / Final         review, video, assembly, output  locked", workspace)
                    self.assertIn("10-15 Video / Final", workspace)
                if stage_id == "03":
                    self.assertNotIn("Pipeline Skills", workspace)

            for stage_id in tuple(f"{index:02d}" for index in range(4, 16)):
                app._select_stage(stage_id)
                workspace = _widget_text(app.query_one("#workspace"))
                if stage_id == "09":
                    self.assertIn("PROMPTS / IMAGE JOBS", workspace)
                elif stage_id == "04":
                    self.assertIn("CREATIVE STRATEGY", workspace)
                    self.assertIn("INPUT CONTEXT", workspace)
                    self.assertIn("STRATEGY BUILD / JSON PREVIEW", workspace)
                    self.assertIn("STRATEGY READOUT / OUTPUT SUMMARY", workspace)
                    self.assertIn("creative_strategy.json", workspace)
                elif stage_id == "05":
                    self.assertIn("BEAT / HOOK PLANNER", workspace)
                    self.assertIn("HOOK OPTIONS / BEAT CANDIDATES", workspace)
                    self.assertIn("SELECTED BEAT PLAN", workspace)
                    self.assertIn("OUTPUT PREVIEW / HANDOFF", workspace)
                    self.assertIn("beat_hook_plan.json", workspace)
                    self.assertNotIn("planned visual", workspace)
                    self.assertNotIn("concept frame", workspace)
                elif stage_id == "06":
                    self.assertIn("CREATIVE JUDGE", workspace)
                    self.assertIn("JUDGE INPUT", workspace)
                    self.assertIn("CREATIVE CHECKS", workspace)
                    self.assertIn("FINAL CREATIVE DECISION", workspace)
                    self.assertIn("OUTPUT PREVIEW / RISKS / HANDOFF", workspace)
                elif stage_id == "07":
                    self.assertIn("SCENE CONTRACTS", workspace)
                    self.assertIn("CURRENT POSITION", workspace)
                    self.assertIn("Status", workspace)
                    self.assertNotIn("HANDOFF PATH", workspace)
                    self.assertNotIn("06 Creative Judge ->", workspace)
                elif stage_id == "08":
                    self.assertIn("PROMPT COMPILER", workspace)
                    self.assertIn("IMAGE COMPILER", workspace)
                    self.assertIn("SCENE CONTRACT INPUTS", workspace)
                    self.assertIn("SCENE PROMPT SUMMARIES", workspace)
                    self.assertIn("FINAL PROMPT PAYLOAD", workspace)
                    self.assertIn("VIDEO COMPILER", workspace)
                    self.assertIn("AUDIO COMPILER", workspace)
                    self.assertIn("MUSIC COMPILER", workspace)
                    self.assertNotIn("Input Readiness", workspace)
                    self.assertNotIn("Output Readiness", workspace)
                    self.assertNotIn("A) ", workspace)
                    self.assertNotIn("B) ", workspace)
                    self.assertNotIn("C) ", workspace)
                    self.assertNotIn("MODEL RULES", workspace)
                    self.assertNotIn("ARTIFACT POLICY", workspace)
                    self.assertNotIn("COMPILER FAMILY / BRANCHES", workspace)
                    self.assertNotIn("OUTPUT / NEXT", workspace)
                    self.assertNotIn("|", workspace)
                    self.assertIn("╭─ IMAGE COMPILER (ACTIVE)", workspace)
                    self.assertNotIn("╭─ SCENE CONTRACT INPUTS", workspace)
                    self.assertNotIn("╭─ SCENE PROMPT SUMMARIES", workspace)
                    self.assertNotIn("╭─ FINAL PROMPT PAYLOAD", workspace)
                    pipeline = _widget_text(app.query_one("#pipeline-map"))
                    self.assertIn("▸ 08 Image Prompt Compiler", pipeline)
                    self.assertIn("○ 12 LTX Video Generation", pipeline)
                else:
                    self.assertIn(STAGE_DEFINITIONS[int(stage_id)].title.upper(), workspace)
                    self.assertIn("Current Status", workspace)
                    self.assertIn("Expected Output", workspace)
                    self.assertIn("Next Action", workspace)

            app._select_stage("09")
            workspace = _widget_text(app.query_one("#workspace"))
            self.assertIn("CURRENT POSITION", workspace)
            self.assertIn("PROMPTS / IMAGE JOBS", workspace)
            self.assertIn("Image 1 / scene_01", workspace)
            self.assertIn("Image 2 / scene_02", workspace)
            self.assertIn("Image 3 / scene_03", workspace)
            self.assertIn("v", workspace)

    async def test_stage09_image_jobs_expand_and_keyboard_selection(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test() as pilot:
            workspace = _widget_text(app.query_one("#workspace"))
            self.assertIn("Image 1 / scene_01", workspace)
            self.assertIn("Image 2 / scene_02", workspace)
            self.assertIn("Image 3 / scene_03", workspace)
            self.assertIn("Preview: keyframes/scene_01.png", workspace)
            self.assertIn("[work] preview", workspace)
            self.assertIn("[empty] slot", workspace)
            self.assertIn("ready", workspace)
            self.assertIn("generating", workspace)
            self.assertIn("in queue", workspace)
            self.assertIn("62%", workspace)
            self.assertIn("00:18 demo", workspace)
            self.assertIn("████", workspace)
            self.assertIn("░░", workspace)
            self.assertNotIn("#", workspace)

            self.assertEqual(2, app.state.selected_image_job)
            await pilot.press("j")
            self.assertEqual(3, app.state.selected_image_job)
            workspace = _widget_text(app.query_one("#workspace"))
            self.assertIn("▸ [empty] slot", workspace)
            await pilot.press("space")
            workspace = _widget_text(app.query_one("#workspace"))
            self.assertIn("Image 3", workspace)
            self.assertIn("waiting", workspace)
            self.assertNotIn("IMAGE JOB 01", workspace)
            self.assertNotIn("Enter/Space expands", workspace)
            self.assertNotIn("····", workspace)
            self.assertNotIn("negative_prompt", workspace)

    async def test_stage_detail_panels_cover_non_image_stages(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test():
            expected = {
                "04": ("STRATEGY READOUT", "Hook", "Core Idea", "Director"),
                "05": ("BEAT / HOOK PLAN", "Beats", "Escalation", "Payoff"),
                "06": ("CREATIVE JUDGE", "Decision", "Rationale", "Selected"),
                "07": ("SCENE CONTRACTS", "scene_count", "Environment", "Text/Glyph Risk"),
                "08": ("PROMPT COMPILER", "IMAGE COMPILER", "SCENE PROMPT SUMMARIES", "VIDEO COMPILER", "AUDIO COMPILER", "MUSIC COMPILER"),
                "10": ("KEYFRAME REVIEW", "Reviewed", "Passed", "Reviewer"),
                "11": ("LTX MOTION PROMPT COMPILER", "Motion Prompts", "Audit", "Render Started"),
                "12": ("LTX VIDEO GENERATION", "Takes Manifest", "Render", "Gate"),
                "13": ("VIDEO REVIEW", "Review Artifact", "Findings", "Reviewer"),
                "14": ("FINAL ASSEMBLY", "Final MP4", "Voice", "Subtitles"),
                "15": ("FINAL OUTPUT", "Final Verdict", "Final MP4", "Postable"),
            }
            for stage_id, needles in expected.items():
                app._select_stage(stage_id)
                workspace = _widget_text(app.query_one("#workspace"))
                self.assertNotIn("Detailed panel planned", workspace)
                self.assertNotIn("PROMPTS / IMAGE JOBS", workspace)
                for needle in needles:
                    self.assertIn(needle, workspace)

    async def test_missing_real_run_is_read_only_and_does_not_crash(self) -> None:
        missing_job = "definitely-missing-run"
        default_root = Path("/workspace/agent_runs")
        missing_path = default_root / missing_job
        self.assertFalse(missing_path.exists())

        app = CreativeOSCockpitApp(CockpitArgs(job_id=missing_job, runs_root=default_root))
        async with app.run_test():
            self.assertEqual("missing", app.state.session_mode)
            self.assertEqual(default_root / missing_job, app.state.data_source_path)
            self.assertFalse(app.state.run_found)
            workspace = _widget_text(app.query_one("#workspace"))
            self.assertIn("Run not found", workspace)
            self.assertIn(f"searched: {default_root / missing_job}", workspace)
            self.assertIn("use --runs-root for fixture/demo data or create a real run first", workspace)
            self.assertIn("missing", _widget_text(app.query_one("#header-meta")))

        self.assertFalse(missing_path.exists())

    def test_state_adapter_sets_real_run_and_watch_flags_without_writing(self) -> None:
        adapter = CockpitStateAdapter(
            job_id="definitely-missing-run",
            runs_root="/workspace/agent_runs",
            watch_enabled=True,
            refresh_sec=2,
        )
        state = adapter.load()
        self.assertEqual("missing", state.session_mode)
        self.assertEqual("missing", state.run_type)
        self.assertEqual(Path("/workspace/agent_runs/definitely-missing-run"), state.data_source_path)
        self.assertTrue(state.watch_enabled)
        self.assertEqual(2, state.refresh_sec)
        self.assertEqual("on / 2s", state.header.watch)
        self.assertFalse(state.data_source_path.exists())

    async def test_agent_core_run_root_artifacts_are_mapped_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            run_dir = runs_root / "cockpit-agent-core-smoke"
            run_dir.mkdir()
            (run_dir / "logs").mkdir()
            self._write_json(run_dir / "result.json", {"job_id": "cockpit-agent-core-smoke", "success": True, "final_phase": "planned"})
            self._write_json(run_dir / "state.json", {"job_id": "cockpit-agent-core-smoke", "status": "planned", "current_phase": "planned"})
            self._write_json(
                run_dir / "plan.json",
                {
                    "job_id": "cockpit-agent-core-smoke",
                    "orientation": "portrait",
                    "width": 512,
                    "height": 768,
                    "target_duration_sec": 3,
                    "selected_pipeline": "ti2vid",
                    "director_output": {"mode": "rule_based_fallback", "llm_active": False},
                },
            )
            self._write_json(
                run_dir / "scene_plan.json",
                {
                    "job_id": "cockpit-agent-core-smoke",
                    "scene_count": 1,
                    "scenes": [{"scene_id": "scene_01", "description": "compact agent-core scene"}],
                },
            )
            self._write_json(
                run_dir / "director_output.json",
                {
                    "director_mode": "rule_based_fallback",
                    "director_llm_active": False,
                },
            )
            self._write_json(
                run_dir / "checkpoints.json",
                {
                    "checkpoints": {
                        "validate_job": {"checkpoint_id": "validate_job", "status": "passed"},
                        "create_prompts": {"checkpoint_id": "create_prompts", "status": "passed"},
                    }
                },
            )
            for name in ("model_prompts.json", "prompt_audit.json", "decision_log.json", "stage_contracts.json"):
                self._write_json(run_dir / name, {"ok": True})
            (run_dir / "logs" / "agent.log").write_text("read-only fixture\n", encoding="utf-8")
            before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

            app = CreativeOSCockpitApp(CockpitArgs(job_id="cockpit-agent-core-smoke", runs_root=runs_root, watch=True, refresh_sec=2))
            async with app.run_test():
                self.assertEqual("agent_core", app.state.run_type)
                self.assertEqual("real_run", app.state.session_mode)
                self.assertTrue(app.state.run_found)
                self.assertEqual(run_dir, app.state.data_source_path)
                self.assertEqual("on / 2s", app.state.header.watch)
                self.assertEqual("agent_core", app.state.header.run_type)
                self.assertIn("DIRECTOR FALLBACK", app.state.issues.blocking_issues)
                self.assertIn("director_llm_active=false", app.state.issues.blocking_issues)
                self.assertIn("final.mp4 missing", app.state.issues.blocking_issues)
                self.assertEqual("warning", app.state.issues.severity)
                self.assertEqual("ok", app.state.skill_health.status)
                self.assertEqual(0, app.state.skill_health.fallback_count)
                self.assertTrue(app.query_one("#issues-tile").has_class("issues-warning"))
                self.assertIn("✓ result.json", [line for line, _ok in app.state.artifacts.lines])
                self.assertIn("○ final.mp4", [line for line, _ok in app.state.artifacts.lines])
                workspace = _widget_text(app.query_one("#workspace"))
                self.assertIn("CURRENT POSITION", workspace)
                self.assertIn("PROMPTS / IMAGE JOBS", workspace)
                self.assertNotIn("PIPELINE PATH", workspace)
                self.assertNotIn("PIPELINE FLOW", workspace)
                self.assertIn("Prompt:", workspace)
                self.assertIn("scene_01", workspace)
                self.assertIn("compact agent-core scene", workspace)
                self.assertIn("FINAL MP4", workspace)
                self.assertIn("○ missing", workspace)
                self.assertIn("DIRECTOR MODE", workspace)
                self.assertIn("rule_based_fallback", workspace)
                self.assertNotIn("READY FOR LTX", workspace)
                self.assertIn("agent_core", _widget_text(app.query_one("#header-meta")))

            after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
            self.assertEqual(before, after)

    async def test_agent_core_final_mp4_present_is_shown_in_active_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            run_dir = runs_root / "cockpit-agent-core-final"
            run_dir.mkdir()
            self._write_json(run_dir / "result.json", {"job_id": "cockpit-agent-core-final", "success": True, "final_phase": "assembled"})
            self._write_json(run_dir / "state.json", {"job_id": "cockpit-agent-core-final", "status": "done", "current_phase": "done"})
            self._write_json(
                run_dir / "plan.json",
                {
                    "job_id": "cockpit-agent-core-final",
                    "selected_pipeline": "ti2vid",
                    "director_output": {"mode": "llm_augmented", "llm_active": True},
                },
            )
            self._write_json(
                run_dir / "scene_plan.json",
                {
                    "scene_count": 1,
                    "scenes": [{"scene_id": "scene_01", "description": "simple completed scene"}],
                },
            )
            self._write_json(
                run_dir / "director_output.json",
                {
                    "director_mode": "llm_augmented",
                    "director_llm_active": True,
                },
            )
            self._write_json(run_dir / "model_prompts.json", {"scenes": [{"scene_id": "scene_01", "model_prompt": "short model prompt summary"}]})
            (run_dir / "final.mp4").write_bytes(b"fake mp4 marker for read-only UI test")

            app = CreativeOSCockpitApp(CockpitArgs(job_id="cockpit-agent-core-final", runs_root=runs_root))
            async with app.run_test():
                workspace = _widget_text(app.query_one("#workspace"))
                self.assertIn("✓ present", workspace)
                self.assertIn("FINAL MP4", workspace)
                self.assertIn("llm_augmented", workspace)
                self.assertIn("short model prompt summary", workspace)
                self.assertEqual(tuple(), app.state.issues.blocking_issues)
                self.assertEqual("none", app.state.issues.severity)
                self.assertEqual("ok", app.state.skill_health.status)

    def test_panel_registry_imports_default_modules(self) -> None:
        expected = {
            "system_status",
            "pipeline_map",
            "active_workspace",
            "skill_health",
            "artifacts",
            "issues",
            "next",
        }
        self.assertEqual(expected, set(PANEL_REGISTRY))
        self.assertEqual(expected | {"header"}, set(PANEL_CONFIG))
        self.assertEqual(expected, set(enabled_panels()))
        for panel_id, panel in PANEL_REGISTRY.items():
            self.assertEqual(panel_id, panel.panel_id)
            self.assertTrue(panel.title)
            self.assertTrue(panel.purpose)
            self.assertTrue(panel.required_data)
            self.assertFalse(panel.optional and panel_id not in {"issues"})

        self.assertTrue(callable(header_panel.render_brand))
        self.assertTrue(callable(system_status_panel.render))
        self.assertTrue(callable(pipeline_map_panel.render))
        self.assertTrue(callable(active_workspace_panel.render))
        self.assertTrue(callable(skill_health_panel.render))
        self.assertTrue(callable(artifacts_panel.render))
        self.assertTrue(callable(issues_panel.render))
        self.assertTrue(callable(next_panel.render))

    async def test_textual_cockpit_theme_colors_are_loaded(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test():
            self.assertEqual("#050B12", app.screen.styles.background.hex)
            self.assertEqual("#050B12", app.query_one("#app-root").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#cockpit-header").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#sidebar").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#system-status").styles.background.hex)
            self.assertEqual("#0B1628", app.query_one("#workspace").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#skill-tile").styles.background.hex)
            self.assertEqual("#38BDF8", app.query_one("#system-status").styles.border.top[1].hex)
            self.assertEqual("#1E3A5F", app.query_one("#pipeline-map").styles.border.top[1].hex)
            self.assertEqual("none", app.state.issues.severity)
            self.assertTrue(app.query_one("#issues-tile").has_class("issues-none"))
            self.assertEqual("#38BDF8", app.query_one("#issues-tile").styles.border.top[1].hex)

    async def test_missing_run_issues_are_error_severity(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="definitely-missing-run", runs_root=Path("/workspace/agent_runs")))
        async with app.run_test():
            self.assertEqual("error", app.state.issues.severity)
            self.assertTrue(app.query_one("#issues-tile").has_class("issues-error"))
            self.assertEqual("#EF4444", app.query_one("#issues-tile").styles.border.top[1].hex)

    async def test_theme_preview_does_not_need_run_data(self) -> None:
        app = ThemePreviewApp()
        async with app.run_test():
            self.assertEqual("#050B12", app.query_one("#theme-preview-root").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#theme-preview-header").styles.background.hex)
            self.assertEqual("#0B1628", app.query_one("#theme-preview-workspace").styles.background.hex)
            self.assertIn("Scene Card Sample", _widget_text(app.query_one("#theme-preview-workspace")))

    def test_scene_card_long_motion_stays_inside_fixed_width(self) -> None:
        card = _scene_card_text(
            {
                "scene_id": "scene_01_with_extra_identifier",
                "keyframe": "/tmp/very/deep/path/scene_01.png",
                "summary": (
                    "slow canopy push-in with sunrise haze, stable jungle frame, parallax leaves, "
                    "controlled suspense forward glide, no new creature, no sudden camera roll, "
                    "and enough extra description to exceed a single card row"
                ),
            }
        )
        lines = str(card).splitlines()
        self.assertGreaterEqual(len(lines), 6)
        self.assertTrue(all(len(line) == 68 for line in lines))
        self.assertTrue(all(line.endswith(("│", "╮", "╯")) for line in lines))
        self.assertIn("motion:", str(card))
        self.assertIn("…", str(card))
        self.assertNotIn("ETA", str(card))
        self.assertNotIn("%", str(card))

    def test_agent_core_long_scene_summary_is_shortened_in_scene_card(self) -> None:
        card = _scene_card_text(
            {
                "scene_id": "scene_01",
                "keyframe": "agent-core scene plan",
                "summary": (
                    "This agent-core scene plan contains an intentionally long prompt summary with many "
                    "details about camera placement, subject motion, environment continuity, lighting, "
                    "and output constraints that should be compacted inside the fixed scene card."
                ),
                "state_label": "READ-ONLY PLAN",
                "status": "final output missing",
            }
        )
        lines = str(card).splitlines()
        self.assertTrue(all(len(line) == 68 for line in lines))
        self.assertIn("…", str(card))
        self.assertIn("READ-ONLY PLAN", str(card))
        self.assertIn("final output missing", str(card))

    def _write_json(self, path: Path, data: object) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")


def _widget_text(widget: object) -> str:
    if getattr(widget, "id", None) == "workspace":
        query_one = getattr(widget, "query_one", None)
        if callable(query_one):
            return _widget_text(query_one("#workspace-content"))
    renderable = getattr(widget, "renderable", None)
    if renderable is not None:
        return str(renderable)
    render = getattr(widget, "render", None)
    if callable(render):
        return str(render())
    return str(widget)


if __name__ == "__main__":
    unittest.main()
