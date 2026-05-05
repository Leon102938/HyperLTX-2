import json
import unittest

from agent_core.feedback_policy import FeedbackAction, evaluate_feedback_actions, suggest_feedback_actions


class G8FeedbackActionsTest(unittest.TestCase):
    def test_feedback_action_serializes_and_deserializes(self) -> None:
        action = FeedbackAction(
            action_id="feedback_001_visible_text",
            issue_type="visible_text",
            action_type="regenerate_keyframe",
            target_stage="storyboard",
            target_scene_id="scene_02",
            target_take_id=None,
            reason="visible_text in frame",
            suggested_fix="regenerate without readable text",
            blocking=True,
            retry_budget_impact="spend_keyframe_retry",
            confidence=0.91,
            source_review_provider="heuristic",
            source_review_real_vlm=False,
            related_checkpoint_id="feedback_review",
        )
        payload = json.loads(json.dumps(action.to_dict()))
        restored = FeedbackAction.from_dict(payload)
        self.assertEqual(restored.action_id, action.action_id)
        self.assertEqual(restored.issue_type, "visible_text")
        self.assertTrue(restored.blocking)

    def test_required_fields_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            FeedbackAction(
                action_id="",
                issue_type="visible_text",
                action_type="regenerate_keyframe",
                target_stage="storyboard",
                target_scene_id=None,
                target_take_id=None,
                reason="visible_text",
                suggested_fix="fix",
                blocking=True,
                retry_budget_impact="spend_keyframe_retry",
                confidence=0.9,
                source_review_provider="heuristic",
                source_review_real_vlm=False,
                related_checkpoint_id="feedback_review",
            )

    def test_unknown_issue_falls_back_to_human_review(self) -> None:
        actions = suggest_feedback_actions({"issues": ["unmapped_weird_issue"]})
        self.assertEqual(actions[0].action_type, "human_review")
        self.assertEqual(actions[0].target_stage, "quality_review")

    def test_each_required_issue_maps_to_expected_action(self) -> None:
        expected = {
            "visible_text": "regenerate_keyframe",
            "fake_text": "regenerate_keyframe",
            "typography": "regenerate_keyframe",
            "phone": "regenerate_keyframe",
            "ui": "regenerate_keyframe",
            "screen": "regenerate_keyframe",
            "app": "regenerate_keyframe",
            "website": "regenerate_keyframe",
            "boring_scene": "choose_alternate_beat_candidate",
            "dead_static_scene": "choose_alternate_beat_candidate",
            "no_visual_change": "choose_alternate_beat_candidate",
            "weak_hook": "choose_alternate_beat_candidate",
            "unclear_action": "simplify_scene",
            "generic_stock_feel": "replan_scene",
            "physical_incoherence": "simplify_scene",
            "low_phone_size_readability": "tighten_prompt",
            "voice_visual_mismatch": "replan_scene",
            "bad_composition": "regenerate_keyframe",
        }
        for issue, action_type in expected.items():
            with self.subTest(issue=issue):
                action = suggest_feedback_actions({"issues": [issue]})[0]
                self.assertEqual(action.action_type, action_type)
                self.assertTrue(action.suggested_fix)
        self.assertTrue(suggest_feedback_actions({"issues": ["visible_text"]})[0].blocking)
        self.assertFalse(suggest_feedback_actions({"issues": ["boring_scene"]})[0].blocking)

    def test_scene_specific_actions_are_created_and_no_issues_passes(self) -> None:
        evaluation = evaluate_feedback_actions(
            {
                "scene_reviews": [
                    {"scene_id": "scene_01", "issues": ["boring_scene"]},
                    {"scene_id": "scene_02", "issues": ["visible_text"]},
                ],
                "provider": "heuristic",
                "real_vlm_inference_used": False,
            },
            {},
            {},
        )
        self.assertEqual(len(evaluation["feedback_actions"]), 2)
        self.assertEqual(evaluation["top_priority_action"].target_scene_id, "scene_02")
        self.assertFalse(evaluation["top_priority_action"].source_review_real_vlm)
        empty = evaluate_feedback_actions({}, {}, {})
        self.assertEqual(empty["feedback_actions"], [])
        self.assertEqual(empty["recommended_next_stage"], "pass")


if __name__ == "__main__":
    unittest.main()
