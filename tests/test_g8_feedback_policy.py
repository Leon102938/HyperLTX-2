import json
import unittest

from agent_core.feedback_policy import evaluate_feedback_actions, suggest_feedback_actions


class G8FeedbackPolicyTest(unittest.TestCase):
    def test_visible_text_maps_to_regenerate_or_reject(self) -> None:
        actions = suggest_feedback_actions({"issues": ["visible_text in frame"]})
        self.assertEqual(actions[0].action_type, "regenerate_keyframe")
        self.assertTrue(actions[0].blocking)

    def test_boring_scene_maps_to_replan(self) -> None:
        actions = suggest_feedback_actions({"creative_quality_warnings": ["boring_scene"]})
        self.assertIn(actions[0].action_type, {"replan_scene", "choose_alternate_beat_candidate"})
        self.assertEqual(actions[0].target_stage, "beat_plan")

    def test_low_phone_size_readability_maps_to_composition_fix_and_serializes(self) -> None:
        actions = suggest_feedback_actions({"platform_fit_warnings": ["low_phone_size_readability"]})
        payload = json.loads(json.dumps([action.to_dict() for action in actions]))
        self.assertEqual(payload[0]["action_type"], "tighten_prompt")
        self.assertEqual(payload[0]["target_stage"], "model_prompting")
        self.assertIn("larger subject", payload[0]["suggested_fix"])

    def test_mixed_issues_prioritize_visible_text_before_boring(self) -> None:
        evaluation = evaluate_feedback_actions(
            {
                "creative_quality_warnings": ["boring_scene"],
                "scene_reviews": [{"scene_id": "scene_02", "issues": ["visible_text"]}],
                "provider": "heuristic",
                "real_vlm_inference_used": False,
            },
            {},
            {},
        )
        self.assertEqual(evaluation["top_priority_action"].issue_type, "visible_text")
        self.assertTrue(evaluation["should_block"])
        self.assertFalse(evaluation["source_review_real_vlm"])


if __name__ == "__main__":
    unittest.main()
