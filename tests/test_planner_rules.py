import unittest

from pydantic import ValidationError

from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, JobInput
from agent_core.adapters.base import VideoAdapter, VoiceAdapter
from agent_core.utils import quantize_duration_to_frame_contract


class PlannerVoiceAdapter(VoiceAdapter):
    name = "planner_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="voice", available=True, phase1_enabled=True)

    def generate_voice(self, job, plan, workspace):  # pragma: no cover
        raise NotImplementedError


class PlannerVideoAdapter(VideoAdapter):
    name = "planner_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            supported_pipelines=["ti2vid", "a2vid"],
        )

    def generate_video(self, job, plan, workspace, voice_result=None):  # pragma: no cover
        raise NotImplementedError


class PlannerRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BackendRegistry([PlannerVoiceAdapter(), PlannerVideoAdapter()])
        self.planner = ProductionPlanner(self.registry)

    def test_voice_duration_drives_target_video_duration(self) -> None:
        job = JobInput(
            idea="An AI agent writes and renders a teaser.",
            script="This is a longer narration that should force the planner to keep the video length aligned with voice.",
            duration_sec=5,
            use_voice=True,
            orientation="portrait",
            resolution="standard",
        )

        initial_plan = self.planner.build_plan(job)
        updated_plan = self.planner.build_plan(job, actual_voice_duration_sec=12.4)

        self.assertEqual(initial_plan.selected_pipeline, "ti2vid")
        self.assertEqual(updated_plan.selected_pipeline, "ti2vid")
        self.assertGreaterEqual(initial_plan.target_duration_sec, initial_plan.estimated_voice_duration_sec + 1.0)
        self.assertGreaterEqual(updated_plan.target_duration_sec, 13.4)
        self.assertEqual((updated_plan.width, updated_plan.height), (704, 1216))
        self.assertIn("Actual voice duration", " ".join(updated_plan.rules_applied))
        self.assertTrue(all((scene.num_frames - 1) % 8 == 0 for scene in updated_plan.scenes))
        quantized_frames, quantized_duration = quantize_duration_to_frame_contract(13.4, 24)
        self.assertGreaterEqual(updated_plan.scenes[0].num_frames, 17)
        self.assertGreaterEqual(updated_plan.target_duration_sec, quantized_duration)

    def test_quantized_duration_contract_stays_stable_for_adapter(self) -> None:
        job = JobInput(
            idea="A compact teaser.",
            duration_sec=4.2,
            use_voice=False,
            orientation="landscape",
            resolution="draft",
        )

        plan = self.planner.build_plan(job)
        video_step = next(step for step in plan.steps if step.name == "video")
        self.assertEqual(video_step.params["num_frames"], 105)
        self.assertAlmostEqual(plan.target_duration_sec, 4.375, places=3)

        recomputed_frames, recomputed_duration = quantize_duration_to_frame_contract(plan.target_duration_sec, 24)
        self.assertEqual(recomputed_frames, 105)
        self.assertAlmostEqual(recomputed_duration, plan.target_duration_sec, places=3)

    def test_phase1_skips_music_and_storyboard(self) -> None:
        job = JobInput(
            idea="Agent teaser",
            use_voice=False,
            use_music=True,
            use_storyboard=True,
            resolution="draft",
            orientation="landscape",
        )

        plan = self.planner.build_plan(job)
        skipped = {step.name: step for step in plan.steps if not step.enabled}
        self.assertEqual(plan.selected_pipeline, "ti2vid")
        self.assertIn("music", skipped)
        self.assertIn("storyboard", skipped)
        self.assertTrue(plan.warnings)

    def test_a2vid_is_rejected_in_phase1(self) -> None:
        job = JobInput(
            idea="Agent teaser",
            script="System online.",
            use_voice=True,
            pipeline_preference="a2vid",
        )

        with self.assertRaisesRegex(ValueError, "not contract-stable"):
            self.planner.build_plan(job)

    def test_invalid_job_input_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            JobInput(idea="", script="", resolution="tiny")

        with self.assertRaises(ValidationError):
            JobInput(idea="x", script="", orientation="diagonal")

        with self.assertRaises(ValidationError):
            JobInput(idea="x", script="", duration_sec=0)


if __name__ == "__main__":
    unittest.main()
