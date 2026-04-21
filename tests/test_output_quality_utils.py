import unittest
from types import SimpleNamespace

from agent_core.utils import (
    build_scene_subtitle_entries,
    compress_visual_prompt,
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


if __name__ == "__main__":
    unittest.main()
