import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.agent_core_cli import _inspect_run


class G8CliFeedbackInspectTest(unittest.TestCase):
    def test_inspect_run_prints_feedback_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir)
            (run / "state.json").write_text(json.dumps({"job_id": "g8-cli", "status": "done", "steps": {}}), encoding="utf-8")
            (run / "result.json").write_text(json.dumps({"success": True, "job_id": "g8-cli", "artifacts": []}), encoding="utf-8")
            (run / "feedback_actions.json").write_text(
                json.dumps(
                    {
                        "recommended_next_stage": "storyboard",
                        "top_priority_action": {
                            "action_id": "feedback_001_visible_text",
                            "issue_type": "visible_text",
                            "action_type": "regenerate_keyframe",
                            "target_stage": "storyboard",
                            "target_scene_id": "scene_02",
                            "blocking": True,
                            "suggested_fix": "regenerate without readable text",
                        },
                        "feedback_actions": [],
                    }
                ),
                encoding="utf-8",
            )
            (run / "retry_plan.json").write_text(
                json.dumps(
                    {
                        "blocked": True,
                        "requires_human_approval": True,
                        "allowed_next_actions": ["regenerate_keyframe"],
                        "invalidated_artifacts": ["keyframes/scene_02"],
                    }
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = _inspect_run(run, tail_lines=5, show_log_tail=False, verbose=False)
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("FEEDBACK", text)
            self.assertIn("regenerate_keyframe", text)
            self.assertIn("regenerate without readable text", text)


if __name__ == "__main__":
    unittest.main()
