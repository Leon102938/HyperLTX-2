import unittest

from agent_core.adapters.base import StoryboardAdapter, VideoAdapter, VoiceAdapter
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


class ScenePlannerStoryboardAdapter(StoryboardAdapter):
    name = "scene_storyboard"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="storyboard", available=True, phase1_enabled=True)

    def generate_storyboard(self, job, plan, workspace):  # pragma: no cover
        raise NotImplementedError


class ScenePlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BackendRegistry(
            [ScenePlannerVoiceAdapter(), ScenePlannerVideoAdapter(), ScenePlannerStoryboardAdapter()]
        )
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
        self.assertEqual(plan.metadata["selection_mode"], "quality_guarded_best_valid_take")
        self.assertEqual(plan.metadata["max_quality_retries_per_scene"], 1)
        self.assertTrue(all(len(scene.takes) == 3 for scene in plan.scenes))
        self.assertTrue(all(scene.takes[0].seed != scene.takes[1].seed for scene in plan.scenes))

    def test_variations_are_generated_and_takes_reference_them(self) -> None:
        job = JobInput(
            idea="A teaser with controlled shot variation.",
            script="First beat. Second beat. Third beat. Fourth beat.",
            duration_sec=10,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"scene_count": 2, "variations_per_scene": 3, "takes_per_scene": 2},
        )

        plan = self.planner.build_plan(job)

        self.assertEqual(plan.metadata["variations_per_scene"], 3)
        self.assertEqual(plan.metadata["takes_per_variation"], 2)
        self.assertEqual(plan.metadata["takes_per_scene"], 6)
        self.assertEqual(plan.steps[-1].params["variations_per_scene"], 3)
        self.assertEqual(plan.steps[-1].params["takes_per_variation"], 2)
        self.assertEqual(plan.steps[-1].params["takes_per_scene"], 6)
        self.assertEqual(plan.metadata["creative_selection_mode"], "rule_based_scene_variation_heuristic")
        self.assertTrue(plan.metadata["creative_selection_enabled"])
        for scene in plan.scenes:
            self.assertEqual(len(scene.variations), 3)
            self.assertEqual(len(scene.takes), 6)
            variation_ids = {variation.variation_id for variation in scene.variations}
            self.assertEqual(len({variation.shot_type for variation in scene.variations}), 3)
            self.assertTrue(all(take.variation_id in variation_ids for take in scene.takes))
            self.assertTrue(all(take.prompt_variant_text for take in scene.takes))
            self.assertTrue(all(take.camera_style or take.camera_motion for take in scene.takes))

    def test_take_render_plan_keeps_selected_variation_context(self) -> None:
        job = JobInput(
            idea="A teaser with variation-aware render plan.",
            script="First beat. Second beat.",
            duration_sec=8,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"scene_count": 1, "variations_per_scene": 2, "takes_per_scene": 2, "force_single_scene": True},
        )

        plan = self.planner.build_plan(job)
        scene = plan.scenes[0]
        take = scene.takes[-1]

        take_plan = self.planner.build_take_render_plan(plan, scene, take)

        self.assertEqual(take_plan.steps[0].params["variation_id"], take.variation_id)
        self.assertEqual(take_plan.steps[0].params["variation_index"], take.variation_index)
        self.assertEqual(take_plan.steps[0].params["creative_selection_mode"], "rule_based_scene_variation_heuristic")
        self.assertEqual(len(take_plan.scenes[0].variations), 1)
        self.assertEqual(take_plan.scenes[0].variations[0].variation_id, take.variation_id)
        self.assertEqual(take_plan.scenes[0].prompt_text, take.prompt_text)

    def test_storyboard_candidates_are_planned_with_preferred_variation(self) -> None:
        job = JobInput(
            idea="A teaser with storyboard planning.",
            script="Scene one opens on the pod. Scene two shows render progress.",
            duration_sec=8,
            use_voice=False,
            use_storyboard=True,
            resolution="draft",
            orientation="landscape",
            metadata={"scene_count": 2, "variations_per_scene": 2, "storyboard_candidates_per_scene": 2},
        )

        plan = self.planner.build_plan(job)

        self.assertTrue(plan.metadata["storyboard_enabled"])
        self.assertEqual(plan.metadata["storyboard_candidates_per_scene"], 2)
        self.assertTrue(plan.steps[1].enabled)
        for scene in plan.scenes:
            self.assertIsNotNone(scene.storyboard_config)
            self.assertTrue(scene.storyboard_config.enabled)
            self.assertEqual(len(scene.keyframe_candidates), 2)
            self.assertEqual(scene.storyboard_config.preferred_variation_id, scene.keyframe_candidates[0].variation_id)
            self.assertTrue(all(candidate.prompt_text for candidate in scene.keyframe_candidates))

    def test_storyboard_render_plan_keeps_candidate_context(self) -> None:
        job = JobInput(
            idea="A teaser with storyboard render plan.",
            script="Scene one opens on the pod.",
            duration_sec=6,
            use_voice=False,
            use_storyboard=True,
            resolution="draft",
            orientation="landscape",
            metadata={"force_single_scene": True, "variations_per_scene": 2, "storyboard_candidates_per_scene": 2},
        )

        plan = self.planner.build_plan(job)
        scene = plan.scenes[0]
        candidate = scene.keyframe_candidates[0]

        storyboard_plan = self.planner.build_storyboard_render_plan(plan, scene, candidate)

        self.assertEqual(storyboard_plan.steps[0].name, "storyboard")
        self.assertEqual(storyboard_plan.steps[0].params["candidate_id"], candidate.candidate_id)
        self.assertEqual(storyboard_plan.steps[0].params["variation_id"], candidate.variation_id)
        self.assertEqual(storyboard_plan.scenes[0].keyframe_candidates[0].candidate_id, candidate.candidate_id)
        self.assertEqual(storyboard_plan.prompt_text, candidate.prompt_text)


if __name__ == "__main__":
    unittest.main()
