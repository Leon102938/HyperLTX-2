from __future__ import annotations

import unittest
from pathlib import Path

from agent_core.creative_os.textual_cockpit import CockpitArgs, CreativeOSCockpitApp


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


if __name__ == "__main__":
    unittest.main()
