import json
import unittest

from agent_core.creative_system import analyze_creative_intent, load_creative_system
from agent_core.schemas import JobInput


class G7CreativeIntentTest(unittest.TestCase):
    def test_intent_is_serializable_non_empty_and_no_script_literal(self) -> None:
        system = load_creative_system()
        job = JobInput(
            job_id="g7-intent",
            idea="Morning Reset: a calm productivity reset.",
            script="Open soft light. Place one clear glass. Breathe by the window.",
            use_voice=False,
            orientation="portrait",
        )
        intent = analyze_creative_intent(
            job=job,
            mode=system.mode("morning_reset"),
            style=system.style("clean_lifestyle_morning"),
            mode_id="morning_reset",
            style_id="clean_lifestyle_morning",
        ).to_dict()
        payload = json.loads(json.dumps(intent))
        self.assertEqual(payload["topic"], "calm morning productivity")
        self.assertTrue(payload["content_promise"])
        self.assertNotIn("Open soft light", payload["sanitized_visual_intent"])
        self.assertNotIn("Place one clear glass", payload["sanitized_visual_intent"])
        self.assertEqual(payload["inferred_mode"], "morning_reset")


if __name__ == "__main__":
    unittest.main()
