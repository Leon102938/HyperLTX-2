import unittest

from agent_core.creative_system import (
    analyze_creative_intent,
    generate_beat_plan_candidates,
    load_creative_system,
    score_beat_plan_candidate,
    select_best_beat_plan_candidate,
)
from agent_core.creative_system.strategy_planner import BeatCandidate, BeatPlanCandidate
from agent_core.schemas import JobInput


class G7BeatPlanCandidateTest(unittest.TestCase):
    def _intent(self):
        system = load_creative_system()
        job = JobInput(
            job_id="g7-candidates",
            idea="Morning Reset: quick calm reset.",
            script="Open soft light. Place one clear glass. Breathe by the window.",
            use_voice=False,
            orientation="portrait",
        )
        return analyze_creative_intent(
            job=job,
            mode=system.mode("morning_reset"),
            style=system.style("clean_lifestyle_morning"),
            mode_id="morning_reset",
            style_id="clean_lifestyle_morning",
        ), system

    def test_generates_three_distinct_candidates_with_actions(self) -> None:
        intent, system = self._intent()
        candidates = generate_beat_plan_candidates(
            creative_intent=intent,
            mode=system.mode("morning_reset"),
            style=system.style("clean_lifestyle_morning"),
            scene_count=3,
        )
        self.assertGreaterEqual(len(candidates), 3)
        self.assertGreaterEqual(len({candidate.hook_pattern for candidate in candidates}), 3)
        self.assertGreaterEqual(len({tuple(candidate.motif_families) for candidate in candidates}), 3)
        for candidate in candidates:
            for beat in candidate.beat_sequence:
                self.assertTrue(beat.shot_recipe_id)
                self.assertTrue(beat.visible_action)
                self.assertTrue(beat.expected_visual_change)
                self.assertNotIn("Open soft light", beat.visible_action)
                self.assertNotIn("Place one clear glass", beat.visible_action)

    def test_scorer_prefers_action_candidate_over_static_generic(self) -> None:
        intent, _ = self._intent()
        good = generate_beat_plan_candidates(creative_intent=intent, scene_count=3)[0]
        bad = BeatPlanCandidate(
            candidate_id="bad_static",
            hook_pattern="weak_hook",
            beat_sequence=[
                BeatCandidate(1, "hook", "generic", "generic_room", "generic_wide", "static generic stock lifestyle room", "", "static camera", ["room"], ["text"], "bad"),
                BeatCandidate(2, "middle", "generic", "generic_room", "generic_wide", "static generic stock lifestyle room", "", "static camera", ["room"], ["text"], "bad"),
                BeatCandidate(3, "payoff", "generic", "generic_room", "generic_wide", "static generic stock lifestyle room", "", "static camera", ["room"], ["text"], "bad"),
            ],
            scene_roles={},
            motif_families=["generic", "generic", "generic"],
            shot_recipes=["generic_wide"],
            continuity_strategy="",
            platform_fit_intent="",
            expected_visual_change=[],
            risk_notes=["text"],
            rationale="generic static stock b-roll",
        )
        self.assertGreater(score_beat_plan_candidate(good, intent)["total_score"], score_beat_plan_candidate(bad, intent)["total_score"])
        selected, scores = select_best_beat_plan_candidate([bad, good], intent)
        self.assertEqual(selected.candidate_id, good.candidate_id)
        self.assertEqual(len(scores), 2)


if __name__ == "__main__":
    unittest.main()
