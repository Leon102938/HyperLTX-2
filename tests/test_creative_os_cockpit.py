from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_core.creative_os.textual_cockpit import CockpitArgs, CreativeOSCockpitApp, ThemePreviewApp, _scene_card_text
from agent_core.creative_os.cockpit.panel_registry import PANEL_CONFIG, PANEL_REGISTRY, enabled_panels
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
            text = app.query_one("#workspace").renderable
            self.assertIn("SCENE JOBS", str(text))
            self.assertIn("scene_01", str(text))
            self.assertIn("scene_03", str(text))
            self.assertIn("READY FOR LTX", str(text))
            brand = str(app.query_one("#header-brand").renderable)
            self.assertIn("CONTENT MASCHINE LIVE", brand)
            self.assertNotIn("▛", brand)
            self.assertIn("fixture/demo", str(app.query_one("#header-meta").renderable))
            self.assertIn("Watch off", str(app.query_one("#keybar").renderable))
            await pilot.press("h")
            self.assertTrue(app.query_one("#help-panel").has_class("visible"))
            await pilot.press("r")
            self.assertEqual("ready_for_ltx_i2v_takes", app.inspection.status)
            self.assertNotIn("/workspace/agent_runs", str(app.inspection.run_dir))

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
            workspace = str(app.query_one("#workspace").renderable)
            self.assertIn("Run not found", workspace)
            self.assertIn(f"searched: {default_root / missing_job}", workspace)
            self.assertIn("use --runs-root for fixture/demo data or create a real run first", workspace)
            self.assertIn("missing", str(app.query_one("#header-meta").renderable))

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
                workspace = str(app.query_one("#workspace").renderable)
                self.assertIn("scene_01", workspace)
                self.assertIn("compact agent-core scene", workspace)
                self.assertIn("AGENT CORE PLAN", workspace)
                self.assertNotIn("READY FOR LTX", workspace)
                self.assertIn("agent_core", str(app.query_one("#header-meta").renderable))

            after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
            self.assertEqual(before, after)

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
            self.assertIn("Scene Card Sample", str(app.query_one("#theme-preview-workspace").renderable))

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

    def _write_json(self, path: Path, data: object) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
