import unittest
from pathlib import Path

from agent_core.backend_registry import build_default_registry
from agent_core.creative_system import build_stage_role_contracts, load_creative_system
from agent_core.planner import ProductionPlanner
from agent_core.schemas import JobInput
from agent_core.utils import evaluate_creative_quality_metadata, evaluate_final_quality_verdict


class G6ReviewSkillPolicyTest(unittest.TestCase):
    def test_review_plan_contains_creative_platform_criteria(self) -> None:
        planner = ProductionPlanner(build_default_registry())
        job = JobInput(
            job_id="g6-review-plan",
            idea="Morning Reset: clean shortform.",
            script="Open light. Place glass. Breathe.",
            duration_sec=8,
            use_voice=False,
            orientation="portrait",
            resolution="draft",
            metadata={"scene_count": 3, "variations_per_scene": 1, "takes_per_scene": 1},
        )
        plan = planner.build_plan(job)
        system = load_creative_system()
        contracts = build_stage_role_contracts(
            job=job,
            plan=plan,
            mode=system.mode("morning_reset"),
            style=system.style("clean_lifestyle_morning"),
            loaded_skills=[],
        )
        review_plan = contracts["review_plan"]
        for criterion in ("boring_scene", "weak_hook", "unclear_action", "generic_stock_feel", "low_phone_size_readability"):
            self.assertIn(criterion, review_plan["creative_quality_checks"])
        self.assertIn("low_phone_size_readability", review_plan["platform_fit_checks"])

    def test_heuristic_review_keeps_real_vlm_false_and_serializes_warnings(self) -> None:
        review = evaluate_creative_quality_metadata(
            scene_world_contract={
                "visible_subject": "generic stock lifestyle interior",
                "environment": "empty room",
                "action": "static scene with no movement",
            }
        )
        self.assertFalse(review["real_vlm_inference_used"])
        self.assertIn("dead_static_or_boring_scene_risk", review["creative_quality_warnings"])

    def test_final_quality_verdict_accepts_creative_platform_warnings(self) -> None:
        verdict = evaluate_final_quality_verdict(
            final_output_path=None,
            expected_width=320,
            expected_height=256,
            expected_frame_rate=24,
            expected_duration_sec=4,
            selected_scene_outputs=[
                {
                    "scene_id": "scene_01",
                    "take_visual_review": {
                        "take_visual_review_status": "needs_review",
                        "postability_score": 0.55,
                        "creative_quality_warnings": ["boring_scene"],
                        "platform_fit_warnings": ["low_phone_size_readability"],
                    },
                }
            ],
        )
        self.assertIn("scene_01: boring_scene", verdict["creative_quality_warnings"])
        self.assertIn("scene_01: low_phone_size_readability", verdict["platform_fit_warnings"])

    def test_qwen_reviewer_prompt_keeps_json_contract(self) -> None:
        text = Path("agent_core/creative_system/prompts/qwen3_vl_reviewer_system.md").read_text(encoding="utf-8")
        self.assertIn("Return JSON only", text)
        self.assertIn("status, postability_score, issues, warnings, problem_frames, summary", text)
        self.assertIn("low phone-size readability", text)


if __name__ == "__main__":
    unittest.main()
