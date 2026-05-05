import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter
from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, JobInput, ProductionPlan


class NoRenderVideoAdapter(VideoAdapter):
    name = "g7_no_render_video"

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
        raise AssertionError("G7 integration tests must not render")


def _job(job_id: str, metadata: dict | None = None) -> JobInput:
    return JobInput(
        job_id=job_id,
        idea="Morning Reset: a clean shortform about light, water, and focus.",
        script="Open soft light. Place one clear glass. Breathe by the window.",
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


class G7PlannerIntegrationTest(unittest.TestCase):
    def test_clean_shortform_plan_uses_selected_candidate_and_per_scene_direction(self) -> None:
        registry = BackendRegistry([NoRenderVideoAdapter()])
        plan = ProductionPlanner(registry).build_plan(_job("g7-plan"))
        self.assertTrue(plan.metadata["g7_candidate_planning_enabled"])
        self.assertEqual(len(plan.metadata["beat_plan_candidates"]), 3)
        self.assertTrue(plan.metadata["selected_beat_plan_candidate"]["candidate_id"])
        self.assertEqual(len(plan.metadata["per_scene_visual_direction"]), 3)
        scene = plan.scenes[0]
        self.assertEqual(scene.prompt_build_metadata["selected_candidate_id"], plan.metadata["selected_candidate_id"])
        self.assertIn("per_scene_visual_direction", scene.prompt_build_metadata["scene_world_contract"])
        for forbidden in ("Open soft light", "Place one clear glass", "Breathe by the window"):
            self.assertNotIn(forbidden, scene.scene_intent.visual_goal)
            self.assertNotIn(forbidden, scene.prompt_build_metadata["positive_model_prompt"])

    def test_simple_video_remains_laufable_without_g7_candidate_requirement(self) -> None:
        registry = BackendRegistry([NoRenderVideoAdapter()])
        job = _job("simple-video-g7-off", {"pipeline_id": "simple_video_v1", "disable_g7_beat_planner": True})
        plan = ProductionPlanner(registry).build_plan(job)
        self.assertFalse(plan.metadata["g7_candidate_planning_enabled"])
        self.assertIsNone(plan.metadata["selected_beat_plan_candidate"])

    def test_stop_after_smoke_writes_g7_trace_and_no_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([NoRenderVideoAdapter()])
            agent = VideoAgent(
                registry=registry,
                planner=ProductionPlanner(registry),
                state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"),
            )
            result = agent.run_job(_job("g7-beat-planner-stop-after-model-prompts-smoke", {"pipeline_dry_run": True, "stop_after": "model_prompts"}))
            run_dir = Path(tmpdir) / "runs" / "g7-beat-planner-stop-after-model-prompts-smoke"
            self.assertTrue(result.success)
            self.assertFalse((run_dir / "final.mp4").exists())
            stage_contracts = json.loads((run_dir / "stage_contracts.json").read_text())
            model_prompts = json.loads((run_dir / "model_prompts.json").read_text())
            decision_log = json.loads((run_dir / "decision_log.json").read_text())
            self.assertIn("creative_intent", stage_contracts)
            self.assertEqual(len(stage_contracts["beat_plan_candidates"]), 3)
            self.assertTrue(stage_contracts["selected_beat_plan_candidate"]["candidate_id"])
            self.assertEqual(len(stage_contracts["per_scene_visual_direction"]), 3)
            self.assertIn("per_scene_visual_direction", model_prompts)
            decision_ids = {entry["decision_id"] for entry in decision_log["decisions"]}
            self.assertIn("beat_plan_candidate_selection", decision_ids)


if __name__ == "__main__":
    unittest.main()
