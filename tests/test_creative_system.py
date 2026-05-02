import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.backend_registry import build_default_registry
from agent_core.creative_system import load_creative_system
from agent_core.planner import ProductionPlanner
from agent_core.prompt_builder import PromptBuilder
from agent_core.schemas import JobInput


class CreativeSystemTest(unittest.TestCase):
    def _job(self) -> JobInput:
        return JobInput(
            job_id="phase-f1-test",
            idea="Morning Reset: Vorhang öffnen, klares Wasserglas auf leerem Holztisch, ruhig am hellen Fenster atmen, keine Geräte.",
            script="Vorhang auf. Stell ein Glas Wasser ab. Atme ruhig am Fenster.",
            duration_sec=9,
            use_voice=True,
            use_storyboard=True,
            orientation="portrait",
            resolution="draft",
            metadata={"subtitle_mode": "off", "scene_count": 3, "variations_per_scene": 1, "takes_per_scene": 3, "storyboard_candidates_per_scene": 3},
        )

    def test_creative_system_loads_morning_reset_mode_and_style(self) -> None:
        system = load_creative_system()
        self.assertEqual(system.mode("morning_reset")["mode_id"], "morning_reset")
        self.assertEqual(system.style("clean_lifestyle_morning")["style_id"], "clean_lifestyle_morning")
        self.assertIn("shot_recipes", system.libraries)
        self.assertIn("anti_patterns", system.libraries)
        self.assertEqual(system.libraries["shot_recipes"]["water_glass_closeup"]["hook_function"], "tactile_detail_hook")
        self.assertIn("fake_text_overlay", system.libraries["anti_patterns"])

    def test_prompt_compiler_separates_debug_and_model_prompt(self) -> None:
        builder = PromptBuilder()
        contract = {
            "visible_subject": "person gently opens plain fabric curtains in soft morning light",
            "environment": "blank wall, simple bedding, soft window light",
            "action": "person gently opens plain fabric curtains",
            "allowed_props": ["plain curtains", "blank wall"],
            "forbidden_props": ["readable text", "phone", "screen", "UI"],
        }
        debug_prompt = builder.build_debug_prompt(["WORLD / SETTING: Vorhang auf.", "SUBJECT / ACTION: person opens curtains."])
        model_prompt = builder.compile_visual_prompt_for_model(scene_world_contract=contract)
        parts = builder.compile_visual_prompt_parts(scene_world_contract=contract)
        self.assertIn("WORLD / SETTING", debug_prompt)
        self.assertNotIn("WORLD / SETTING", model_prompt)
        self.assertNotIn("Vorhang auf", model_prompt)
        self.assertIn("person gently opens plain fabric curtains", model_prompt)
        self.assertNotIn("phone", parts["positive_model_prompt"].lower())
        self.assertIn("phone", parts["negative_model_prompt"].lower())
        self.assertLessEqual(len(parts["negative_model_terms"]), 25)

    def test_morning_reset_model_prompts_use_playbook_motifs(self) -> None:
        planner = ProductionPlanner(build_default_registry())
        plan = planner.build_plan(self._job())
        self.assertEqual(plan.metadata["mode_id"], "morning_reset")
        motifs = [scene.prompt_build_metadata["scene_world_contract"].get("motif_id") for scene in plan.scenes]
        self.assertEqual(motifs, ["curtain_opening_window_light", "water_glass_empty_table", "calm_breathing_open_window"])
        scene2_meta = plan.scenes[1].prompt_build_metadata
        scene3_meta = plan.scenes[2].prompt_build_metadata
        scene2 = scene2_meta["model_prompt"].lower()
        scene3 = scene3_meta["model_prompt"].lower()
        scene2_positive = scene2_meta["positive_model_prompt"].lower()
        scene2_negative = scene2_meta["negative_model_prompt"].lower()
        scene3_positive = scene3_meta["positive_model_prompt"].lower()
        scene3_negative = scene3_meta["negative_model_prompt"].lower()
        self.assertIn("one clear water glass only", scene2)
        self.assertIn("plain empty wooden table", scene2)
        self.assertNotIn("phone", scene2_positive)
        self.assertNotIn("screen", scene2_positive)
        self.assertNotIn("ui", scene2_positive)
        self.assertIn("phones", scene2_negative)
        self.assertIn("screens", scene2_negative)
        self.assertIn("black rectangle", scene2_negative)
        self.assertLessEqual(len(scene2_positive.split()), 100)
        self.assertLessEqual(len(scene2.split()), 140)
        self.assertIn("single full-frame", scene3_positive)
        self.assertIn("continuous lifestyle shot", scene3_positive)
        self.assertNotIn("no single full-frame", scene3)
        self.assertNotIn("no one continuous", scene3)
        self.assertIn("split screen", scene3_negative)
        self.assertIn("collage", scene3_negative)
        self.assertIn("panels", scene3_negative)
        self.assertEqual(scene2_meta["backend_prompt_policy"]["zimage"], "positive_only")
        self.assertEqual(scene2_meta["backend_prompt_policy"]["ltx"], "positive_plus_short_avoid")
        self.assertEqual(scene2_meta["zimage_prompt_sent"], scene2_meta["positive_model_prompt"])
        self.assertNotIn("Avoid:", scene2_meta["zimage_prompt_sent"])
        self.assertIn("Avoid:", scene2_meta["ltx_prompt_sent"])
        self.assertLessEqual(len(scene2_meta["zimage_prompt_sent"].split()), 100)
        self.assertLessEqual(len(scene2_meta["ltx_prompt_sent"].split()), 140)
        self.assertEqual(scene2_meta["shot_recipe_id"], "water_glass_closeup")
        self.assertEqual(scene2_meta["hook_function"], "tactile_detail_hook")

    def test_storyboard_effective_model_prompt_is_clean(self) -> None:
        planner = ProductionPlanner(build_default_registry())
        plan = planner.build_plan(self._job())
        scene = plan.scenes[0]
        storyboard_plan = planner.build_storyboard_render_plan(plan, scene, scene.keyframe_candidates[0])
        model_prompt = storyboard_plan.steps[0].params["effective_model_prompt"]
        self.assertNotIn("WORLD / SETTING", model_prompt)
        self.assertNotIn("Vorhang auf", model_prompt)
        self.assertNotIn("Stell ein Glas Wasser ab", model_prompt)
        self.assertNotIn("Avoid:", model_prompt)
        self.assertEqual(storyboard_plan.steps[0].params["prompt_sent_to_backend_source"]["zimage"], "positive_model_prompt")

    def test_prompt_audit_file_is_written_for_plan_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VideoAgent(state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"))
            job = self._job()
            state = agent.state_store.initialize(job)
            plan = agent.planner.build_plan(job)
            agent._save_prompt_audit(job, plan, state)
            audit = json.loads((Path(tmpdir) / "runs" / job.job_id / "prompt_audit.json").read_text())
            self.assertTrue(audit["checks"]["no_debug_labels_in_model_prompts"])
            self.assertTrue(audit["checks"]["no_script_snippets_in_model_prompts"])
            self.assertTrue(audit["checks"]["positive_model_prompt_word_count_ok"])
            self.assertTrue(audit["checks"]["positive_model_prompt_no_risky_words"])
            self.assertTrue(audit["checks"]["negative_model_prompt_separate"])
            self.assertTrue(audit["checks"]["no_positive_constraints_in_negative_prompt"])
            self.assertTrue(audit["checks"]["model_prompt_not_overlong"])
            self.assertTrue(audit["checks"]["no_repeated_forbidden_spam"])
            self.assertTrue(audit["checks"]["scene_has_shot_recipe_id"])
            self.assertTrue(audit["checks"]["scene_has_hook_function"])
            self.assertTrue(audit["checks"]["anti_patterns_checked"])
            self.assertTrue(audit["checks"]["backend_prompt_policy_applied"])
            self.assertTrue(audit["checks"]["zimage_positive_only_applied"])
            self.assertTrue(audit["checks"]["ltx_short_avoid_applied"])

    def test_model_prompts_trace_file_is_written_for_plan_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VideoAgent(state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"))
            job = self._job()
            state = agent.state_store.initialize(job)
            plan = agent.planner.build_plan(job)
            agent._save_model_prompts_trace(job, plan, state)
            trace = json.loads((Path(tmpdir) / "runs" / job.job_id / "model_prompts.json").read_text())
            self.assertEqual(trace["backend_prompt_policy"]["zimage"], "positive_only")
            self.assertTrue(trace["checks"]["zimage_positive_only_applied"])
            self.assertTrue(trace["checks"]["ltx_short_avoid_applied"])
            self.assertTrue(trace["checks"]["no_debug_labels_in_backend_prompts"])
            self.assertTrue(trace["checks"]["no_script_snippets_in_backend_prompts"])
            self.assertTrue(trace["checks"]["scene_has_shot_recipe_id"])
            scene2 = trace["scenes"][1]
            self.assertEqual(scene2["prompt_sent_to_backend_source"]["zimage"], "positive_model_prompt")
            self.assertNotIn("Avoid:", scene2["zimage_prompt_sent"])
            self.assertIn("Avoid:", scene2["ltx_prompt_sent"])
            self.assertLessEqual(len(scene2["zimage_prompt_sent"].split()), 100)
            self.assertLessEqual(len(scene2["ltx_prompt_sent"].split()), 140)


if __name__ == "__main__":
    unittest.main()
