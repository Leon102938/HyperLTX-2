import unittest

from agent_core.feedback_policy import (
    RetryBudget,
    build_feedback_checkpoint_state,
    build_retry_plan,
    evaluate_feedback_actions,
)


class G8RetryPlanTest(unittest.TestCase):
    def test_retry_budget_counts_remaining_and_exhausted(self) -> None:
        budget = RetryBudget(used_retries={"keyframe": 1, "video": 0, "plan": 1})
        self.assertEqual(budget.remaining_retries()["keyframe"], 0)
        self.assertEqual(budget.remaining_retries()["video"], 1)
        self.assertFalse(budget.exhausted())
        exhausted = RetryBudget(used_retries={"keyframe": 1, "video": 1, "plan": 1})
        self.assertTrue(exhausted.exhausted())

    def test_retry_plan_invalidates_prompt_changed_artifacts(self) -> None:
        actions = evaluate_feedback_actions({"issues": [{"issue_type": "visible_text", "scene_id": "scene_02"}]}, {}, {})["feedback_actions"]
        plan = build_retry_plan(actions)
        payload = plan.to_dict()
        self.assertTrue(payload["blocked"])
        self.assertIn("model_prompts/scene_02", payload["invalidated_artifacts"])
        self.assertIn("takes/scene_02", payload["invalidated_artifacts"])

    def test_choose_alternate_take_invalidates_no_prompts(self) -> None:
        actions = evaluate_feedback_actions({"issues": [{"issue_type": "boring_scene", "scene_id": "scene_01"}]}, {}, {})["feedback_actions"]
        replacement = actions[0].__class__(
            **{**actions[0].to_dict(), "action_type": "choose_alternate_take", "retry_budget_impact": "choose_existing_take"}
        )
        plan = build_retry_plan([replacement])
        self.assertEqual(plan.invalidated_artifacts, [])
        self.assertIn("existing_valid_takes", plan.reusable_artifacts)

    def test_replan_scene_invalidates_scene_artifacts(self) -> None:
        actions = evaluate_feedback_actions({"issues": [{"issue_type": "generic_stock_feel", "scene_id": "scene_03"}]}, {}, {})["feedback_actions"]
        plan = build_retry_plan(actions)
        self.assertIn("scene_plan/scene_03", plan.invalidated_artifacts)
        self.assertIn("model_prompts/scene_03", plan.invalidated_artifacts)

    def test_exhausted_budget_forces_human_review_checkpoint_state(self) -> None:
        evaluation = evaluate_feedback_actions(
            {"issues": [{"issue_type": "visible_text", "scene_id": "scene_02"}]},
            {},
            {"retry_budget": {"used_retries": {"keyframe": 1, "video": 1, "plan": 1}}},
        )
        self.assertTrue(evaluation["retry_plan"].requires_human_approval)
        checkpoint_state = build_feedback_checkpoint_state(evaluation)
        self.assertEqual(checkpoint_state["blocked_by_feedback_action_id"], "feedback_001_visible_text")
        self.assertTrue(checkpoint_state["feedback_requires_approval"])


if __name__ == "__main__":
    unittest.main()
