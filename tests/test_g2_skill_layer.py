import json
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.backend_registry import build_default_registry
from agent_core.creative_system import load_creative_system
from agent_core.creative_system.skill_loader import load_required_skills, load_skill
from agent_core.decision_log import build_initial_decision_log, decision_entry
from agent_core.pipeline import load_pipeline_definition
from agent_core.planner import ProductionPlanner
from agent_core.prompt_builder import PromptBuilder
from agent_core.schemas import JobInput
from scripts.agent_core_cli import _build_parser, _build_payload


class G2SkillLayerTest(unittest.TestCase):
    def _job(self) -> JobInput:
        return JobInput(
            job_id="g2-skill-test",
            idea="Morning Reset: a quiet clean lifestyle short about starting focused.",
            script="Open the room to soft light. Put one simple object down. Reset your posture and breathe.",
            duration_sec=9,
            use_voice=False,
            use_storyboard=False,
            orientation="portrait",
            resolution="draft",
            metadata={
                "pipeline_id": "clean_shortform_v1",
                "pipeline_dry_run": True,
                "scene_count": 3,
                "variations_per_scene": 1,
                "takes_per_scene": 1,
            },
        )

    def test_skill_loader_loads_markdown_skill(self) -> None:
        skill = load_skill("models/zimage_turbo")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.skill_id, "models/zimage_turbo")
        self.assertIn("positive", skill.purpose.lower())
        self.assertFalse(skill.missing_fields)

    def test_skill_loader_reports_missing_skills(self) -> None:
        result = load_required_skills(["models/zimage_turbo", "missing/nope"])
        self.assertEqual([skill.skill_id for skill in result.loaded], ["models/zimage_turbo"])
        self.assertEqual(result.missing, ["missing/nope"])

    def test_clean_shortform_pipeline_loads_required_skills(self) -> None:
        pipeline = load_pipeline_definition("clean_shortform_v1")
        self.assertEqual(pipeline.pipeline_id, "clean_shortform_v1")
        self.assertIn("platforms/tiktok_shortform", pipeline.required_skills)
        prompt_step = next(step for step in pipeline.steps if step.step_id == "create_prompts")
        self.assertIn("models/zimage_turbo", prompt_step.required_skills)
        self.assertIn("creative_strategy", pipeline.stage_roles)

    def test_prompt_and_model_audit_contains_loaded_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VideoAgent(state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"))
            result = agent.run_job(self._job())
            self.assertTrue(result.success)
            self.assertEqual(result.final_phase, "planned")
            self.assertEqual(result.message, "Pipeline dry-run completed before voice/video/storyboard/model backends.")
            run_dir = Path(tmpdir) / "runs" / "g2-skill-test"
            prompt_audit = json.loads((run_dir / "prompt_audit.json").read_text())
            model_prompts = json.loads((run_dir / "model_prompts.json").read_text())
            self.assertEqual(prompt_audit["pipeline_id"], "clean_shortform_v1")
            self.assertEqual(model_prompts["pipeline_id"], "clean_shortform_v1")
            self.assertIn("platforms/tiktok_shortform", prompt_audit["required_skills"])
            self.assertTrue(any(skill["skill_id"] == "models/zimage_turbo" for skill in model_prompts["loaded_skills"]))
            self.assertEqual(model_prompts["missing_skills"], [])

    def test_model_prompt_policy_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VideoAgent(state_store=__import__("agent_core.state_store", fromlist=["StateStore"]).StateStore(Path(tmpdir) / "runs"))
            agent.run_job(self._job())
            trace = json.loads((Path(tmpdir) / "runs" / "g2-skill-test" / "model_prompts.json").read_text())
            self.assertEqual(trace["backend_prompt_policy"]["zimage"], "positive_only")
            self.assertEqual(trace["backend_prompt_policy"]["ltx"], "positive_plus_short_avoid")
            self.assertFalse(trace["backend_prompt_policy_notes"]["ltx_negative_prompt_supported"])
            scene = trace["scenes"][0]
            self.assertEqual(scene["prompt_sent_to_backend_source"]["zimage"], "positive_model_prompt")
            self.assertIn("ltx_positive_prompt_sent", scene)
            self.assertIn("ltx_negative_prompt_sent", scene)

    def test_morning_reset_uses_motif_families(self) -> None:
        mode = load_creative_system().mode("morning_reset")
        self.assertIn("motif_families", mode)
        self.assertIn("light_reveal", mode["motif_families"])
        self.assertEqual(mode["scene_arc"]["selection_mode"], "flexible_from_motif_families")
        self.assertEqual(mode["shot_recipe_policy"], "recipes are selectable building blocks, not fixed mandatory scenes")

    def test_decision_log_schema_and_utility(self) -> None:
        entry = decision_entry("selected_pipeline", "pipeline", "clean_shortform_v1")
        self.assertEqual(entry.decision_id, "selected_pipeline")
        planner = ProductionPlanner(build_default_registry())
        job = self._job()
        plan = planner.build_plan(job)
        plan.metadata["pipeline_id"] = "clean_shortform_v1"
        plan.metadata["required_skills"] = ["models/zimage_turbo"]
        log = build_initial_decision_log(
            job,
            plan,
            pipeline_id="clean_shortform_v1",
            skill_trace={"required_skills": ["models/zimage_turbo"], "loaded_skills": [], "missing_skills": []},
        )
        self.assertEqual(log.pipeline_id, "clean_shortform_v1")
        self.assertIn("selected_skill_set", [decision.decision_id for decision in log.decisions])

    def test_cli_payload_flags_set_metadata(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--idea",
                "dry run",
                "--script",
                "short script",
                "--pipeline-dry-run",
                "--approval-gates-enabled",
            ]
        )
        payload = _build_payload(args)
        metadata = payload["job"]["metadata"]
        self.assertTrue(metadata["pipeline_dry_run"])
        self.assertTrue(metadata["approval_gates_enabled"])

    def test_zimage_policy_constant_remains_positive_only(self) -> None:
        self.assertEqual(PromptBuilder.DEFAULT_BACKEND_PROMPT_POLICY["zimage"], "positive_only")
        self.assertEqual(PromptBuilder.DEFAULT_BACKEND_PROMPT_POLICY["ltx"], "positive_plus_short_avoid")


if __name__ == "__main__":
    unittest.main()
