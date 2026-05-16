import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter
from agent_core.backend_registry import BackendRegistry
from agent_core.creative_system import build_skill_injection_context, load_creative_system
from agent_core.pipeline import load_pipeline_definition
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, JobInput, ProductionPlan


class NoRenderVideoAdapter(VideoAdapter):
    name = "g6_no_render_video"

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
        raise AssertionError("G6 tests must not render")


def _job(job_id: str = "g6-skill-injection") -> JobInput:
    return JobInput(
        job_id=job_id,
        idea="Morning Reset: a clean shortform about light, water, and calm focus.",
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
        },
    )


class G6SkillInjectionTest(unittest.TestCase):
    def test_context_loads_clean_shortform_skills_and_serializes(self) -> None:
        system = load_creative_system()
        pipeline = load_pipeline_definition("clean_shortform_v1")
        mode = system.mode("morning_reset")
        style = system.style("clean_lifestyle_morning")
        context = build_skill_injection_context(
            pipeline_def=pipeline,
            mode=mode,
            style=style,
            job_metadata={"pipeline_id": "clean_shortform_v1"},
        ).to_dict()
        payload = json.loads(json.dumps(context))
        self.assertEqual(payload["pipeline_id"], "clean_shortform_v1")
        self.assertIn("platforms/tiktok_shortform", payload["required_skills"])
        self.assertIn("models/hidream_o1_dev", payload["model_skills"])
        self.assertIn("directing/anti_boring_scene", payload["directing_skills"])
        self.assertEqual(payload["prompt_policy"]["hidream"], "positive_only")
        self.assertEqual(payload["missing_skills"], [])

    def test_context_reports_missing_skill_without_hard_failure(self) -> None:
        system = load_creative_system()
        pipeline = load_pipeline_definition("simple_video_v1").model_copy(
            update={"required_skills": ["missing/g6_optional_skill"]}
        )
        context = build_skill_injection_context(
            pipeline_def=pipeline,
            mode=system.mode("morning_reset"),
            style=system.style("clean_lifestyle_morning"),
            job_metadata={},
        ).to_dict()
        self.assertIn("missing/g6_optional_skill", context["missing_skills"])
        self.assertIn("missing_optional_or_required_skills_recorded", context["warnings"])

    def test_simple_video_context_remains_backwards_compatible(self) -> None:
        pipeline = load_pipeline_definition("simple_video_v1")
        context = build_skill_injection_context(pipeline_def=pipeline).to_dict()
        self.assertEqual(context["pipeline_id"], "simple_video_v1")
        self.assertEqual(context["required_skills"], [])
        self.assertEqual(context["loaded_skills"], [])
        self.assertEqual(context["missing_skills"], [])

    def test_stop_after_model_prompts_traces_skills_contracts_and_no_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([NoRenderVideoAdapter()])
            agent = VideoAgent(
                registry=registry,
                planner=ProductionPlanner(registry),
                state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"),
            )
            job = _job("g6-skill-injection-stop-after-model-prompts-smoke").model_copy(
                update={"metadata": {**_job().metadata, "stop_after": "model_prompts", "pipeline_dry_run": True}}
            )
            result = agent.run_job(job)
            run_dir = Path(tmpdir) / "runs" / job.job_id
            self.assertTrue(result.success)
            self.assertFalse((run_dir / "final.mp4").exists())
            for name in ("prompt_audit.json", "model_prompts.json", "stage_contracts.json", "decision_log.json"):
                self.assertTrue((run_dir / name).exists(), name)
            audit = json.loads((run_dir / "prompt_audit.json").read_text())
            trace = json.loads((run_dir / "model_prompts.json").read_text())
            decisions = json.loads((run_dir / "decision_log.json").read_text())
            self.assertIn("skill_injection_context", audit)
            self.assertIn("creative_strategy", trace)
            self.assertTrue(trace["stage_contracts"]["creative_strategy"]["success_criteria"])
            decision_ids = {entry["decision_id"] for entry in decisions["decisions"]}
            self.assertIn("creative_strategy_decision", decision_ids)
            self.assertIn("beat_plan_decision", decision_ids)
            self.assertIn("review_plan_decision", decision_ids)


if __name__ == "__main__":
    unittest.main()
