from __future__ import annotations

import unittest
from pathlib import Path

from agent_core.creative_os.textual_cockpit import CockpitArgs, CreativeOSCockpitApp, ThemePreviewApp


FIXTURE_RUNS_ROOT = Path("/workspace/tests/fixtures/creative_os_runs")


class CreativeOSCockpitTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_cockpit_loads_fixture_layout(self) -> None:
        app = CreativeOSCockpitApp(CockpitArgs(job_id="creative-os-jungle-001", runs_root=FIXTURE_RUNS_ROOT))
        async with app.run_test() as pilot:
            self.assertEqual("ready_for_ltx_i2v_takes", app.inspection.status)
            text = app.query_one("#workspace").renderable
            self.assertIn("SCENE JOBS", str(text))
            self.assertIn("scene_01", str(text))
            self.assertIn("READY FOR LTX", str(text))
            self.assertIn("fixture/demo", str(app.query_one("#header-meta").renderable))
            await pilot.press("h")
            self.assertTrue(app.query_one("#help-panel").has_class("visible"))
            await pilot.press("r")
            self.assertEqual("ready_for_ltx_i2v_takes", app.inspection.status)

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

    async def test_theme_preview_does_not_need_run_data(self) -> None:
        app = ThemePreviewApp()
        async with app.run_test():
            self.assertEqual("#050B12", app.query_one("#theme-preview-root").styles.background.hex)
            self.assertEqual("#07111F", app.query_one("#theme-preview-header").styles.background.hex)
            self.assertEqual("#0B1628", app.query_one("#theme-preview-workspace").styles.background.hex)
            self.assertIn("Scene Card Sample", str(app.query_one("#theme-preview-workspace").renderable))


if __name__ == "__main__":
    unittest.main()
