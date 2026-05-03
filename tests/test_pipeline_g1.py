import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter
from agent_core.backend_registry import BackendRegistry
from agent_core.pipeline import checkpoint_for_step, load_pipeline_definition
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, CheckpointRecord, ExecutionResult, JobInput, ProductionPlan
from agent_core.state_store import StateStore


class NoRenderVideoAdapter(VideoAdapter):
    name = "no_render_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            transport="fake",
            supported_pipelines=["ti2vid"],
        )

    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        raise AssertionError("dry-run or blocking approval tests must not call video rendering")


def _agent(tmpdir: str) -> tuple[VideoAgent, StateStore]:
    registry = BackendRegistry([NoRenderVideoAdapter()])
    store = StateStore(Path(tmpdir) / "runs")
    return (
        VideoAgent(
            registry=registry,
            state_store=store,
            planner=ProductionPlanner(registry),
        ),
        store,
    )


def _job_payload(job_id: str, metadata: dict | None = None) -> dict:
    return {
        "job_id": job_id,
        "idea": "A controlled dry-run validates the production machine.",
        "duration_sec": 4,
        "use_voice": False,
        "resolution": "320x256",
        "orientation": "landscape",
        "metadata": metadata or {},
        "backend_overrides": {"director_llm": {"enabled": False}},
    }


class PipelineG1Test(unittest.TestCase):
    def test_pipeline_definition_loading(self) -> None:
        definition = load_pipeline_definition("simple_video_v1")

        self.assertEqual(definition.pipeline_id, "simple_video_v1")
        self.assertEqual([step.step_id for step in definition.steps][:5], [
            "validate_job",
            "create_plan",
            "approve_plan",
            "create_prompts",
            "approve_prompts",
        ])
        self.assertIn("final_quality_gate", definition.checkpoints)
        self.assertEqual(definition.approval_policy.mode, "manual_file")
        self.assertTrue(definition.default_policy["cli_fields_to_show"])

    def test_checkpoint_status_transitions_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir) / "runs")
            job = JobInput.model_validate(_job_payload("checkpoint-transitions"))
            state = store.initialize(job)
            state.pipeline_id = "simple_video_v1"
            definition = load_pipeline_definition("simple_video_v1")

            checkpoint = checkpoint_for_step(definition, "validate_job")
            self.assertEqual(checkpoint.status, "pending")
            store.record_checkpoint(state, checkpoint)
            checkpoint.status = "passed"
            checkpoint.reason = "unit test passed checkpoint"
            store.record_checkpoint(state, checkpoint)

            payload = json.loads((store.job_dir(job.job_id or "") / "checkpoints.json").read_text())
            saved = payload["checkpoints"]["validate_job"]
            self.assertEqual(saved["status"], "passed")
            self.assertEqual(saved["reason"], "unit test passed checkpoint")
            self.assertEqual(payload["current_checkpoint_id"], "validate_job")

    def test_blocking_approval_gate_stops_before_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = _agent(tmpdir)

            result = agent.run_job(
                _job_payload(
                    "approval-blocked",
                    metadata={"approval_gates_enabled": True},
                )
            )

            job_dir = store.job_dir("approval-blocked")
            state = json.loads((job_dir / "state.json").read_text())
            checkpoints = json.loads((job_dir / "checkpoints.json").read_text())
            self.assertFalse(result.success)
            self.assertEqual(result.final_phase, "planned")
            self.assertEqual(state["blocked_by_checkpoint_id"], "approve_plan")
            self.assertEqual(checkpoints["checkpoints"]["approve_plan"]["status"], "needs_review")
            self.assertFalse((job_dir / "final.mp4").exists())
            self.assertIn("approval-blocked/approvals/approve_plan.json", result.metadata["approval_path"])

    def test_local_approval_file_allows_progress_to_next_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = _agent(tmpdir)
            job_dir = store.job_dir("approval-file")
            approvals_dir = job_dir / "approvals"
            approvals_dir.mkdir(parents=True, exist_ok=True)
            (approvals_dir / "approve_plan.json").write_text(
                json.dumps({"approved": True, "approved_by": "unit-test"}),
                encoding="utf-8",
            )

            result = agent.run_job(
                _job_payload(
                    "approval-file",
                    metadata={"approval_gates_enabled": True},
                )
            )

            checkpoints = json.loads((job_dir / "checkpoints.json").read_text())["checkpoints"]
            self.assertFalse(result.success)
            self.assertEqual(checkpoints["approve_plan"]["status"], "passed")
            self.assertEqual(checkpoints["approve_plan"]["approved_by"], "unit-test")
            self.assertEqual(checkpoints["approve_prompts"]["status"], "needs_review")
            self.assertEqual(result.metadata["blocked_checkpoint_id"], "approve_prompts")
            self.assertFalse((job_dir / "final.mp4").exists())

    def test_dry_run_without_real_models_records_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = _agent(tmpdir)

            result = agent.run_job(
                _job_payload(
                    "pipeline-dry-run",
                    metadata={"pipeline_dry_run": True},
                )
            )

            job_dir = store.job_dir("pipeline-dry-run")
            checkpoints = json.loads((job_dir / "checkpoints.json").read_text())["checkpoints"]
            self.assertTrue(result.success)
            self.assertEqual(result.final_phase, "planned")
            self.assertTrue(result.metadata["pipeline_dry_run"])
            self.assertFalse(result.metadata["render_started"])
            self.assertFalse((job_dir / "final.mp4").exists())
            self.assertEqual(checkpoints["validate_job"]["status"], "passed")
            self.assertEqual(checkpoints["create_plan"]["status"], "passed")
            self.assertEqual(checkpoints["approve_plan"]["status"], "passed")
            self.assertEqual(checkpoints["create_prompts"]["status"], "passed")
            self.assertEqual(checkpoints["approve_prompts"]["status"], "passed")
            self.assertTrue((job_dir / "prompt_audit.json").exists())
            self.assertTrue((job_dir / "model_prompts.json").exists())

    def test_state_contains_checkpoints_for_cli_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = _agent(tmpdir)

            agent.run_job(
                _job_payload(
                    "state-checkpoints",
                    metadata={"pipeline_dry_run": True},
                )
            )

            state = json.loads((store.job_dir("state-checkpoints") / "state.json").read_text())
            self.assertEqual(state["pipeline_id"], "simple_video_v1")
            self.assertEqual(state["current_checkpoint_id"], "approve_prompts")
            self.assertIsNone(state["blocked_by_checkpoint_id"])
            self.assertIn("approve_prompts", state["checkpoints"])
            checkpoint = state["checkpoints"]["approve_prompts"]
            self.assertEqual(checkpoint["stage"], "prompt_approval")
            self.assertTrue(checkpoint["approval_required"])
            self.assertEqual(checkpoint["status"], "passed")


if __name__ == "__main__":
    unittest.main()
