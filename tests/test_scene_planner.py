import unittest

from agent_core.adapters.base import VideoAdapter, VoiceAdapter
from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, JobInput


class ScenePlannerVoiceAdapter(VoiceAdapter):
    name = "scene_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="voice", available=True, phase1_enabled=True)

    def generate_voice(self, job, plan, workspace):  # pragma: no cover
        raise NotImplementedError


class ScenePlannerVideoAdapter(VideoAdapter):
    name = "scene_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            supported_pipelines=["ti2vid"],
        )

    def generate_video(self, job, plan, workspace, voice_result=None):  # pragma: no cover
        raise NotImplementedError


class ScenePlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BackendRegistry([ScenePlannerVoiceAdapter(), ScenePlannerVideoAdapter()])
        self.planner = ProductionPlanner(self.registry)

    def test_segmenter_creates_multiple_scenes_for_longer_job(self) -> None:
        job = JobInput(
            idea="A modular rendering core produces a teaser.",
            script="Scene one opens on the pod. Scene two shows planning. Scene three shows rendering. Scene four closes on the final export.",
            duration_sec=12,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
        )

        plan = self.planner.build_plan(job)

        self.assertEqual(plan.metadata["segmentation_mode"], "multi_scene")
        self.assertGreater(len(plan.scenes), 1)
        self.assertAlmostEqual(
            plan.target_duration_sec,
            round(sum(scene.target_duration_sec for scene in plan.scenes), 3),
            places=3,
        )
        self.assertTrue(all(scene.shots for scene in plan.scenes))
        self.assertTrue(all(scene.num_frames > 0 for scene in plan.scenes))
        self.assertTrue(all(len(scene.takes) == 1 for scene in plan.scenes))

    def test_force_single_scene_keeps_phase1_fallback(self) -> None:
        job = JobInput(
            idea="A compact teaser.",
            script="One clean scene only.",
            duration_sec=8,
            use_voice=True,
            resolution="draft",
            orientation="landscape",
            metadata={"force_single_scene": True},
        )

        plan = self.planner.build_plan(job)

        self.assertEqual(plan.metadata["segmentation_mode"], "single_scene")
        self.assertEqual(len(plan.scenes), 1)
        self.assertEqual(plan.steps[-1].params["scene_count"], 1)

    def test_explicit_scene_count_drives_duration_distribution(self) -> None:
        job = JobInput(
            idea="A modular teaser with explicit scene count.",
            script="First beat. Second beat. Third beat. Fourth beat.",
            duration_sec=10,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"scene_count": 3},
        )

        plan = self.planner.build_plan(job)

        self.assertEqual(len(plan.scenes), 3)
        self.assertAlmostEqual(
            plan.target_duration_sec,
            round(sum(scene.target_duration_sec for scene in plan.scenes), 3),
            places=3,
        )
        self.assertTrue(all(scene.target_duration_sec >= 2.0 for scene in plan.scenes))
        self.assertEqual(plan.metadata["scene_count"], 3)

    def test_takes_per_scene_is_planned_and_seeded(self) -> None:
        job = JobInput(
            idea="A teaser with take variation.",
            script="First beat. Second beat. Third beat.",
            duration_sec=10,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"scene_count": 2, "takes_per_scene": 3},
        )

        plan = self.planner.build_plan(job)

        self.assertEqual(plan.metadata["takes_per_scene"], 3)
        self.assertEqual(plan.steps[-1].params["takes_per_scene"], 3)
        self.assertTrue(all(len(scene.takes) == 3 for scene in plan.scenes))
        self.assertTrue(all(scene.takes[0].seed != scene.takes[1].seed for scene in plan.scenes))


if __name__ == "__main__":
    unittest.main()
