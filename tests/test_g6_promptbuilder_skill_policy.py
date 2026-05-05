import unittest

from agent_core.backend_registry import build_default_registry
from agent_core.planner import ProductionPlanner
from agent_core.prompt_builder import PromptBuilder
from agent_core.schemas import JobInput


class G6PromptBuilderSkillPolicyTest(unittest.TestCase):
    def _plan(self):
        planner = ProductionPlanner(build_default_registry())
        job = JobInput(
            job_id="g6-prompt-policy",
            idea="Morning Reset with quiet light, one clear water glass, and calm breathing.",
            script="Open the curtains. Place one clear glass. Breathe by the window.",
            duration_sec=8,
            use_voice=False,
            use_storyboard=False,
            orientation="portrait",
            resolution="draft",
            metadata={"scene_count": 3, "variations_per_scene": 1, "takes_per_scene": 1},
        )
        return planner.build_plan(job)

    def test_zimage_positive_only_has_no_avoid_debug_or_script(self) -> None:
        scene = self._plan().scenes[1]
        meta = scene.prompt_build_metadata
        zimage = meta["zimage_prompt_sent"]
        self.assertEqual(zimage, meta["positive_model_prompt"])
        self.assertNotIn("Avoid:", zimage)
        for label in PromptBuilder.DEBUG_LABELS:
            self.assertNotIn(label, zimage)
        self.assertNotIn("Place one clear glass", zimage)
        self.assertGreaterEqual(len(zimage.split()), 25)
        self.assertLessEqual(len(zimage.split()), 100)

    def test_ltx_positive_and_negative_trace_are_separate(self) -> None:
        scene = self._plan().scenes[1]
        meta = scene.prompt_build_metadata
        self.assertEqual(meta["ltx_positive_prompt_sent"], meta["positive_model_prompt"])
        self.assertEqual(meta["ltx_negative_prompt_sent"], meta["negative_model_prompt"])
        self.assertIn("Avoid:", meta["ltx_prompt_sent"])
        self.assertIn("phones", meta["ltx_negative_prompt_sent"])
        self.assertNotIn("phones", meta["ltx_positive_prompt_sent"].lower())

    def test_model_prompt_plan_fields_are_populated_from_prompt_parts(self) -> None:
        builder = PromptBuilder()
        parts = builder.compile_visual_prompt_parts(
            scene_world_contract={
                "visible_subject": "hand places one clear water glass on a plain empty wooden table",
                "environment": "soft natural morning light on clean wood",
                "action": "hand places one clear water glass",
                "allowed_props": ["one clear water glass only", "plain empty wooden table"],
                "forbidden_props": ["phones", "screens", "text"],
                "backend_prompt_policy": {"zimage": "positive_only", "ltx": "positive_plus_short_avoid"},
            }
        )
        self.assertEqual(parts["zimage_prompt_sent"], parts["positive_model_prompt"])
        self.assertEqual(parts["ltx_positive_prompt_sent"], parts["positive_model_prompt"])
        self.assertEqual(parts["ltx_negative_prompt_sent"], parts["negative_model_prompt"])


if __name__ == "__main__":
    unittest.main()
