import unittest
from types import SimpleNamespace

from agent_core.adapters.zimage_storyboard_adapter import ZImageStoryboardAdapter
from agent_core.schemas import ProductionPlan, ProductionStep
from agent_core.utils import (
    build_scene_subtitle_entries,
    compress_visual_prompt,
    evaluate_keyframe_visual_risk,
    format_overlay_title_text,
    overlay_layout_profile,
)


class OutputQualityUtilsTest(unittest.TestCase):
    def test_compress_visual_prompt_drops_narration_and_sanitizes_text_prone_objects(self) -> None:
        prompt = (
            "Wenn dein Morgen hektisch startet, aendere nicht alles auf einmal. "
            "Hook focus: messy bedroom morning, chaotic alarm clock, shallow depth of field, cinematic lighting, 4k, realistic. "
            "Visual goal: Close-up of a chaotic alarm clock and scattered papers on a messy nightstand, shallow depth of field, warm morning light. "
            "Story beat: opening_hook."
        )

        compressed = compress_visual_prompt(prompt)

        self.assertNotIn("Wenn dein Morgen hektisch startet", compressed)
        self.assertIn("clean tabletop surface", compressed)
        self.assertIn("no letters, no numbers", compressed)

    def test_build_scene_subtitle_entries_merges_or_avoids_one_word_fragments(self) -> None:
        scene = SimpleNamespace(
            narration_text="Dann schreib genau eine wichtige Aufgabe auf, bevor du Nachrichten oeffnest.",
            narration_start_sec=0.0,
            narration_end_sec=4.0,
        )

        entries = build_scene_subtitle_entries(
            [scene],
            max_words=7,
            max_chars=42,
            min_words=2,
            min_chars=8,
            min_segment_duration_sec=1.0,
        )

        self.assertTrue(entries)
        self.assertNotIn("auf", {entry["text"] for entry in entries})
        self.assertTrue(all((entry["end_sec"] - entry["start_sec"]) >= 1.0 for entry in entries))

    def test_overlay_title_formatting_wraps_and_scales_long_titles(self) -> None:
        formatted = format_overlay_title_text(
            "3 kleine Morgen-Gewohnheiten fuer einen ruhigeren und fokussierten Start",
            max_chars_per_line=18,
            max_lines=3,
        )

        lines = formatted.splitlines()
        profile = overlay_layout_profile(formatted)

        self.assertGreater(len(lines), 1)
        self.assertLessEqual(len(lines), 3)
        self.assertGreater(profile["font_divisor"], 20)
        self.assertGreaterEqual(profile["top_margin"], 44)

    def test_zimage_storyboard_adapter_prefers_step_effective_prompt(self) -> None:
        adapter = ZImageStoryboardAdapter()
        plan = ProductionPlan(
            job_id="storyboard-prompt-test",
            orientation="portrait",
            resolution_label="draft",
            width=576,
            height=1024,
            render_profile="balanced",
            selected_pipeline="ti2vid",
            target_duration_sec=4.0,
            prompt_text="global plan prompt that should not be used",
            steps=[
                ProductionStep(
                    name="storyboard",
                    kind="storyboard",
                    enabled=True,
                    params={
                        "effective_prompt": "scene specific keyframe prompt, no readable text, no screens, no paper.",
                        "prompt_source": "scene_world_contract_candidate_variation",
                    },
                )
            ],
        )

        prompt, source = adapter._resolve_effective_prompt(plan, plan.steps[0])

        self.assertEqual(prompt, "scene specific keyframe prompt, no readable text, no screens, no paper.")
        self.assertEqual(source, "scene_world_contract_candidate_variation")

    def test_zimage_storyboard_adapter_falls_back_to_compressed_global_prompt(self) -> None:
        adapter = ZImageStoryboardAdapter()
        plan = ProductionPlan(
            job_id="storyboard-prompt-fallback",
            orientation="portrait",
            resolution_label="draft",
            width=576,
            height=1024,
            render_profile="balanced",
            selected_pipeline="ti2vid",
            target_duration_sec=4.0,
            prompt_text="Camera: person by window, clean morning light, calm portrait frame.",
            steps=[ProductionStep(name="storyboard", kind="storyboard", enabled=True, params={})],
        )

        prompt, source = adapter._resolve_effective_prompt(plan, plan.steps[0])

        self.assertEqual(source, "global_plan_prompt_compressed")
        self.assertIn("person by window", prompt)
        self.assertIn("no visible text", prompt)

    def test_keyframe_visual_risk_allows_forbidden_policy_words(self) -> None:
        review = evaluate_keyframe_visual_risk(
            scene_world_contract={
                "visible_subject": "person opening curtains in soft morning light",
                "environment": "tidy bedroom with clean unlabeled surfaces",
                "action": "opening curtains",
                "allowed_props": ["glass of water", "curtains", "window light"],
                "forbidden_props": ["paper", "screens", "readable text", "logos"],
                "text_risk_policy": "No readable text, no paper, no screens, no logos.",
                "social_tip_visual_guard": True,
            },
            candidate_prompt_text="Forbidden visuals: paper, screens, readable text. Text risk policy: no paper, no screens.",
            effective_prompt="No readable text, no screens, no paper, no typography, no glyphs.",
        )

        self.assertEqual(review["visual_risk_status"], "passed")
        self.assertEqual(review["issues"], [])

    def test_keyframe_visual_risk_rejects_positive_allowed_or_action_risk(self) -> None:
        review = evaluate_keyframe_visual_risk(
            scene_world_contract={
                "visible_subject": "person at desk",
                "environment": "workspace",
                "action": "writing on paper",
                "allowed_props": ["open notebook", "visible screen"],
                "forbidden_props": ["paper", "screens"],
                "text_risk_policy": "No readable text.",
            },
            candidate_prompt_text="Close-up of paper on the desk with readable text.",
            effective_prompt="Scene keyframe: person writing on paper.",
        )

        self.assertEqual(review["visual_risk_status"], "rejected")
        self.assertTrue(any("allowed_props" in issue for issue in review["issues"]))
        self.assertTrue(any("action" in issue for issue in review["issues"]))

    def test_keyframe_visual_risk_missing_contract_needs_review(self) -> None:
        review = evaluate_keyframe_visual_risk(
            scene_world_contract={"visible_subject": "person by window"},
            candidate_prompt_text="Clean portrait keyframe.",
            effective_prompt="Clean portrait keyframe, no readable text.",
        )

        self.assertEqual(review["visual_risk_status"], "needs_review")
        self.assertTrue(review["warnings"])


if __name__ == "__main__":
    unittest.main()
