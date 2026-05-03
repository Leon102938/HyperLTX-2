import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter
from agent_core.backend_registry import BackendRegistry
from agent_core.creative_system.contracts import build_stage_role_contracts
from agent_core.pipeline import normalize_stop_after
from agent_core.planner import ProductionPlanner
from agent_core.resume_contract import inspect_resume_contract
from agent_core.schemas import BackendCapabilities, JobInput, ProductionPlan
from agent_core.utils import evaluate_creative_quality_metadata, evaluate_final_quality_verdict
from scripts.agent_core_cli import _build_parser, _build_payload


class NoRenderVideoAdapter(VideoAdapter):
    name = "g345_no_render_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            transport="fake",
            supported_pipelines=["ti2vid"],
        )

    def generate_video(self, job: JobInput, plan: ProductionPlan, workspace: Path, voice_result=None):
        raise AssertionError("G3/G4/G5 architecture tests must not render")


def _agent(tmpdir: str) -> VideoAgent:
    registry = BackendRegistry([NoRenderVideoAdapter()])
    return VideoAgent(registry=registry, state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"), planner=ProductionPlanner(registry))


def _job(job_id: str, metadata: dict | None = None) -> JobInput:
    return JobInput(
        job_id=job_id,
        idea="Morning Reset: a controlled clean shortform plan.",
        script="Open soft light, place one simple object, and reset posture.",
        duration_sec=8,
        use_voice=False,
        use_storyboard=False,
        orientation="portrait",
        resolution="draft",
        metadata={
            "pipeline_id": "clean_shortform_v1",
            "scene_count": 3,
            "variations_per_scene": 1,
            "takes_per_scene": 1,
            **(metadata or {}),
        },
    )


class G3G4G5ArchitectureTest(unittest.TestCase):
    def test_stage_role_contracts_serialize(self) -> None:
        planner = ProductionPlanner(BackendRegistry([NoRenderVideoAdapter()]))
        job = _job("contracts-serialize")
        plan = planner.build_plan(job)
        contracts = build_stage_role_contracts(job=job, plan=plan, mode={}, style={}, loaded_skills=[])
        payload = json.loads(json.dumps(contracts))
        self.assertIn("creative_strategy", payload)
        self.assertIn("beat_plan", payload)
        self.assertIn("visual_direction", payload)
        self.assertIn("model_prompt_plan", payload)
        self.assertIn("review_plan", payload)
        self.assertIn("creative_quality_checks", payload["review_plan"])

    def test_prompt_audit_contains_stage_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _agent(tmpdir)
            agent.run_job(_job("stage-contract-trace", {"stop_after": "model_prompts"}))
            run_dir = Path(tmpdir) / "runs" / "stage-contract-trace"
            audit = json.loads((run_dir / "prompt_audit.json").read_text())
            trace = json.loads((run_dir / "model_prompts.json").read_text())
            self.assertIn("stage_contracts", audit)
            self.assertEqual(audit["stage_contracts"]["contract_version"], "g3_stage_role_contracts_v1")
            self.assertIn("review_plan", trace["stage_contracts"])
            self.assertTrue((run_dir / "stage_contracts.json").exists())

    def test_cli_payload_sets_stop_after(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--idea", "x", "--script", "y", "--stop-after", "model_prompts"])
        payload = _build_payload(args)
        self.assertEqual(payload["job"]["metadata"]["stop_after"], "model_prompts")
        self.assertEqual(normalize_stop_after(payload["job"]["metadata"]), "model_prompts")

    def test_stop_after_model_prompts_starts_no_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _agent(tmpdir)
            result = agent.run_job(_job("stop-after-model-prompts", {"stop_after": "model_prompts"}))
            run_dir = Path(tmpdir) / "runs" / "stop-after-model-prompts"
            self.assertTrue(result.success)
            self.assertEqual(result.metadata["stopped_after"], "model_prompts")
            self.assertFalse(result.metadata["render_started"])
            self.assertTrue((run_dir / "prompt_audit.json").exists())
            self.assertTrue((run_dir / "model_prompts.json").exists())
            self.assertFalse((run_dir / "final.mp4").exists())

    def test_resume_contract_detects_rejection_and_reusable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "resume-contract"
            run_dir.mkdir(parents=True)
            (run_dir / "plan.json").write_text("{}", encoding="utf-8")
            (run_dir / "model_prompts.json").write_text("{}", encoding="utf-8")
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "current_checkpoint_id": "approve_prompts",
                        "blocked_by_checkpoint_id": "approve_prompts",
                        "checkpoints": {
                            "approve_prompts": {
                                "status": "needs_review",
                                "approval_required": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            approval_dir = run_dir / "approvals"
            approval_dir.mkdir()
            (approval_dir / "approve_prompts.json").write_text(json.dumps({"approved": False, "approved_by": "unit"}), encoding="utf-8")
            contract = inspect_resume_contract(run_dir)
            self.assertFalse(contract["resume_supported"])
            self.assertTrue(contract["has_rejection"])
            self.assertFalse(contract["can_continue_by_contract"])
            self.assertTrue(contract["reusable_artifacts"]["plan"]["exists"])
            self.assertTrue(contract["reusable_artifacts"]["model_prompts"]["exists"])

    def test_creative_quality_review_is_metadata_only(self) -> None:
        review = evaluate_creative_quality_metadata(
            scene_world_contract={
                "visible_subject": "generic stock lifestyle interior",
                "environment": "empty room",
                "action": "static scene with no movement",
            }
        )
        self.assertEqual(review["provider"], "heuristic_metadata")
        self.assertFalse(review["real_vlm_inference_used"])
        self.assertIn("dead_static_or_boring_scene_risk", review["creative_quality_warnings"])

    def test_final_quality_verdict_accepts_creative_and_platform_warnings(self) -> None:
        verdict = evaluate_final_quality_verdict(
            final_output_path=None,
            expected_width=320,
            expected_height=256,
            expected_frame_rate=24,
            expected_duration_sec=4,
            selected_scene_outputs=[
                {
                    "scene_id": "scene_01",
                    "take_visual_review_status": "needs_review",
                    "postability_score": 0.55,
                    "take_visual_review": {
                        "take_visual_review_status": "needs_review",
                        "postability_score": 0.55,
                        "creative_quality_warnings": ["boring_scene"],
                        "platform_fit_warnings": ["weak_hook"],
                    },
                }
            ],
        )
        self.assertIn("scene_01: boring_scene", verdict["creative_quality_warnings"])
        self.assertIn("scene_01: weak_hook", verdict["platform_fit_warnings"])
        self.assertIn("creative_quality_metadata_review", verdict["quality_sources"])


if __name__ == "__main__":
    unittest.main()
