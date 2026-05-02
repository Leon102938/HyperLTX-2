import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path


def _load_cli_module():
    path = Path("/workspace/scripts/agent_core_cli.py")
    spec = importlib.util.spec_from_file_location("agent_core_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class CliLiveDashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = _load_cli_module()

    def test_live_flag_policy(self) -> None:
        self.assertTrue(self.cli._should_use_live(Namespace(live=True, no_live=False)))
        self.assertFalse(self.cli._should_use_live(Namespace(live=True, no_live=True)))

    def test_live_dashboard_includes_prompt_policy(self) -> None:
        lines = self.cli._format_live_lines(
            {"job_id": "job-1", "status": "running", "current_phase": "storyboard"},
            {"job_id": "job-1", "status": "running", "current_phase": "storyboard", "steps": {}},
            {},
            {},
            run_dir=None,
            base_url="http://127.0.0.1:8000",
            start=0.0,
            phase_started=0.0,
            quiet=False,
            verbose=False,
        )
        joined = "\n".join(lines)
        self.assertIn("CONTENT MASCHINE LIVE", joined)
        self.assertIn("CURRENT PROMPT", joined)
        self.assertIn("Policy", joined)


if __name__ == "__main__":
    unittest.main()
