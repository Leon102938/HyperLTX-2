import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.agent_core_cli import (
    _inspect_checkpoints,
    _inspect_run,
    render_checkpoint_summary,
    write_checkpoint_decision,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _checkpoint(checkpoint_id: str, *, status: str = "needs_review") -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "stage": "plan_approval",
        "status": status,
        "blocking": True,
        "reason": "unit test gate",
        "issues": ["issue one"],
        "warnings": ["warning one"],
        "related_artifacts": [
            {"key": "plan_file", "kind": "json", "path": "/workspace/agent_runs/test/plan.json", "origin": "test"}
        ],
        "approval_required": True,
        "approved_by": None,
        "approved_at": None,
        "metadata": {"approval_path": "/workspace/agent_runs/test/approvals/approve_plan.json"},
    }


def _run_with_checkpoints(root: Path, *, checkpoints_file: bool = True) -> Path:
    run_dir = root / "sample-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": "sample-run",
        "pipeline_id": "simple_video_v1",
        "current_checkpoint_id": "approve_plan",
        "blocked_by_checkpoint_id": "approve_plan",
        "checkpoints": {"approve_plan": _checkpoint("approve_plan")},
    }
    state = {
        "job_id": "sample-run",
        "status": "planned",
        "current_phase": "planned",
        "pipeline_id": "simple_video_v1",
        "current_checkpoint_id": "approve_plan",
        "blocked_by_checkpoint_id": "approve_plan",
        "checkpoints": payload["checkpoints"],
    }
    if checkpoints_file:
        _write_json(run_dir / "checkpoints.json", payload)
    _write_json(run_dir / "state.json", state)
    _write_json(
        run_dir / "result.json",
        {
            "job_id": "sample-run",
            "success": False,
            "final_phase": "planned",
            "message": "waiting for approval",
            "metadata": {"approval_blocked": True},
        },
    )
    return run_dir


class CliCheckpointTest(unittest.TestCase):
    def test_checkpoint_inspect_reads_checkpoints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_with_checkpoints(Path(tmpdir), checkpoints_file=True)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = _inspect_checkpoints(run_dir)

            output = buffer.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("INSPECT CHECKPOINTS", output)
            self.assertIn("approve_plan", output)
            self.assertIn("Source", output)
            self.assertIn("checkpoints.json", output)
            self.assertIn("--approve-checkpoint", output)

    def test_checkpoint_inspect_falls_back_to_state_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_with_checkpoints(Path(tmpdir), checkpoints_file=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                shown = render_checkpoint_summary(run_dir)

            output = buffer.getvalue()
            self.assertTrue(shown)
            self.assertIn("state.json", output)
            self.assertIn("Blocked by", output)

    def test_approval_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_with_checkpoints(Path(tmpdir))

            path = write_checkpoint_decision(
                run_dir,
                "approve_plan",
                approved=True,
                actor="human",
                note="looks good",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path, run_dir / "approvals" / "approve_plan.json")
            self.assertTrue(payload["approved"])
            self.assertEqual(payload["approved_by"], "human")
            self.assertEqual(payload["note"], "looks good")

    def test_reject_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_with_checkpoints(Path(tmpdir))

            path = write_checkpoint_decision(
                run_dir,
                "approve_plan",
                approved=False,
                actor="human",
                note="not good enough",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["approved"])
            self.assertEqual(payload["approved_by"], "human")
            self.assertEqual(payload["note"], "not good enough")

    def test_approval_file_does_not_escape_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "escape-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_id = "../../escape"
            _write_json(
                run_dir / "checkpoints.json",
                {
                    "job_id": "escape-run",
                    "pipeline_id": "simple_video_v1",
                    "current_checkpoint_id": checkpoint_id,
                    "blocked_by_checkpoint_id": checkpoint_id,
                    "checkpoints": {checkpoint_id: _checkpoint(checkpoint_id)},
                },
            )

            with self.assertRaises(RuntimeError):
                write_checkpoint_decision(
                    run_dir,
                    checkpoint_id,
                    approved=True,
                    actor="human",
                    note="bad path",
                )

            self.assertFalse((Path(tmpdir) / "escape.json").exists())

    def test_existing_inspect_run_stays_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_with_checkpoints(Path(tmpdir))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = _inspect_run(run_dir, tail_lines=0, show_log_tail=False, verbose=False)

            output = buffer.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("RUN FAILED", output)
            self.assertIn("CHECKPOINTS", output)


if __name__ == "__main__":
    unittest.main()
