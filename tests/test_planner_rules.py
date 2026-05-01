import unittest
import re

from pydantic import ValidationError

from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import BackendCapabilities, JobInput
from agent_core.adapters.base import MusicBackendAdapter, VideoAdapter, VoiceAdapter
from agent_core.utils import quantize_duration_to_frame_contract


class PlannerVoiceAdapter(VoiceAdapter):
    name = "planner_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="voice", available=True, phase1_enabled=True)

    def generate_voice(self, job, plan, workspace):  # pragma: no cover
        raise NotImplementedError


class PlannerVideoAdapter(VideoAdapter):
    name = "planner_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            supported_pipelines=["ti2vid", "a2vid"],
        )

    def generate_video(self, job, plan, workspace, voice_result=None):  # pragma: no cover
        raise NotImplementedError


class PlannerMusicAdapter(MusicBackendAdapter):
    name = "planner_music"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="music", available=True, phase1_enabled=True)

    def generate_music(self, job, plan, workspace, voice_result=None):  # pragma: no cover
        raise NotImplementedError


class PlannerRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BackendRegistry([PlannerVoiceAdapter(), PlannerVideoAdapter()])
        self.planner = ProductionPlanner(self.registry)

    def test_voice_duration_drives_target_video_duration(self) -> None:
        job = JobInput(
            idea="An AI agent writes and renders a teaser.",
            script="This is a longer narration that should force the planner to keep the video length aligned with voice.",
            duration_sec=5,
            use_voice=True,
            orientation="portrait",
            resolution="standard",
        )

        initial_plan = self.planner.build_plan(job)
        updated_plan = self.planner.build_plan(job, actual_voice_duration_sec=12.4)

        self.assertEqual(initial_plan.selected_pipeline, "ti2vid")
        self.assertEqual(updated_plan.selected_pipeline, "ti2vid")
        self.assertGreaterEqual(initial_plan.target_duration_sec, initial_plan.estimated_voice_duration_sec + 1.0)
        self.assertGreaterEqual(updated_plan.target_duration_sec, 13.4)
        self.assertEqual((updated_plan.width, updated_plan.height), (704, 1216))
        self.assertIn("Actual voice duration", " ".join(updated_plan.rules_applied))
        self.assertTrue(all((scene.num_frames - 1) % 8 == 0 for scene in updated_plan.scenes))
        quantized_frames, quantized_duration = quantize_duration_to_frame_contract(13.4, 24)
        self.assertGreaterEqual(updated_plan.scenes[0].num_frames, 17)
        self.assertGreaterEqual(updated_plan.target_duration_sec, quantized_duration)

    def test_quantized_duration_contract_stays_stable_for_adapter(self) -> None:
        job = JobInput(
            idea="A compact teaser.",
            duration_sec=4.2,
            use_voice=False,
            orientation="landscape",
            resolution="draft",
        )

        plan = self.planner.build_plan(job)
        video_step = next(step for step in plan.steps if step.name == "video")
        self.assertEqual(video_step.params["num_frames"], 105)
        self.assertAlmostEqual(plan.target_duration_sec, 4.375, places=3)

        recomputed_frames, recomputed_duration = quantize_duration_to_frame_contract(plan.target_duration_sec, 24)
        self.assertEqual(recomputed_frames, 105)
        self.assertAlmostEqual(recomputed_duration, plan.target_duration_sec, places=3)

    def test_phase1_skips_music_and_storyboard(self) -> None:
        job = JobInput(
            idea="Agent teaser",
            use_voice=False,
            use_music=True,
            use_storyboard=True,
            resolution="draft",
            orientation="landscape",
        )

        plan = self.planner.build_plan(job)
        skipped = {step.name: step for step in plan.steps if not step.enabled}
        self.assertEqual(plan.selected_pipeline, "ti2vid")
        self.assertIn("music", skipped)
        self.assertIn("storyboard", skipped)
        self.assertTrue(plan.warnings)

    def test_music_is_enabled_when_backend_is_available(self) -> None:
        registry = BackendRegistry([PlannerVoiceAdapter(), PlannerVideoAdapter(), PlannerMusicAdapter()])
        plan = ProductionPlanner(registry).build_plan(
            JobInput(
                idea="Agent teaser",
                use_voice=False,
                use_music=True,
                resolution="draft",
                orientation="landscape",
            )
        )

        music_step = next(step for step in plan.steps if step.name == "music")
        self.assertTrue(music_step.enabled)
        self.assertEqual(music_step.adapter_name, "planner_music")
        self.assertFalse(any("Music requested but skipped" in warning for warning in plan.warnings))

    def test_a2vid_is_rejected_in_phase1(self) -> None:
        job = JobInput(
            idea="Agent teaser",
            script="System online.",
            use_voice=True,
            pipeline_preference="a2vid",
        )

        with self.assertRaisesRegex(ValueError, "not contract-stable"):
            self.planner.build_plan(job)

    def test_invalid_job_input_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            JobInput(idea="", script="", resolution="tiny")

        with self.assertRaises(ValidationError):
            JobInput(idea="x", script="", orientation="diagonal")

        with self.assertRaises(ValidationError):
            JobInput(idea="x", script="", duration_sec=0)

    def test_social_tip_visual_guard_avoids_text_prone_scene_motifs(self) -> None:
        job = JobInput(
            idea="Drei Morgen-Gewohnheiten fuer einen fokussierten Start.",
            script=(
                "Wenn dein Morgen hektisch startet, aendere nicht alles auf einmal. "
                "Leg zuerst dein Handy ausser Reichweite und trink direkt ein Glas Wasser. "
                "Dann schreib genau eine wichtige Aufgabe auf, bevor du Nachrichten oeffnest. "
                "Drei kleine Schritte, weniger Reibung, mehr Klarheit fuer den Rest des Tages."
            ),
            duration_sec=22,
            use_voice=True,
            use_storyboard=True,
            use_music=True,
            orientation="portrait",
            resolution="draft",
            metadata={"subtitle_mode": "burn", "variations_per_scene": 4},
        )

        plan = self.planner.build_plan(job)

        self.assertTrue(plan.metadata["social_tip_visual_guard"])
        self.assertEqual(plan.metadata["social_tip_visual_guard_family"], "morning_reset")
        self.assertEqual(plan.metadata["subtitle_min_words"], 3)
        self.assertAlmostEqual(plan.metadata["subtitle_min_duration_sec"], 1.1)
        guarded_text = []
        for scene in plan.scenes:
            if not scene.scene_intent:
                continue
            guarded_text.extend(
                [
                    scene.scene_intent.hook_focus,
                    scene.scene_intent.visual_goal,
                    scene.scene_intent.shot_intent,
                ]
            )
        flattened = " ".join(guarded_text).lower()
        self.assertNotIn("writing on notepad", flattened)
        self.assertNotIn("handwriting", flattened)
        self.assertNotIn("paper", flattened)
        self.assertIn("window", flattened)
        self.assertIn("water", flattened)

    def test_social_tip_visual_guard_focus_break_family_uses_stable_motifs(self) -> None:
        job = JobInput(
            idea="Vier kurze Focus-Break Moves fuer mehr Klarheit im Arbeitstag.",
            script=(
                "Wenn dein Kopf am Bildschirm dicht macht, steh kurz auf. "
                "Trink erst einen Schluck Wasser, roll die Schultern und atme am Fenster tief durch. "
                "Dann geh mit ruhigerem Blick zurueck an den Tisch."
            ),
            duration_sec=18,
            use_voice=True,
            use_storyboard=True,
            use_music=True,
            orientation="portrait",
            resolution="draft",
            metadata={"subtitle_mode": "burn", "variations_per_scene": 4},
        )

        plan = self.planner.build_plan(job)

        self.assertTrue(plan.metadata["social_tip_visual_guard"])
        self.assertEqual(plan.metadata["social_tip_visual_guard_family"], "focus_break")
        self.assertIn("workspace turned away from displays", plan.director_output.style_lock.visual_identity.lower())
        self.assertNotIn("screen", plan.director_output.style_lock.visual_identity.lower())
        flattened = " ".join(
            f"{scene.scene_intent.hook_focus} {scene.scene_intent.visual_goal} {scene.scene_intent.shot_intent}"
            for scene in plan.scenes
            if scene.scene_intent
        ).lower()
        self.assertIn("stretch", flattened)
        self.assertIn("window", flattened)
        self.assertIn("water", flattened)
        self.assertNotIn("paper", flattened)
        self.assertNotIn("screen", flattened)
        self.assertNotIn("notebook", flattened)

    def test_prompt_builder_v2_emits_structured_scene_world_contract(self) -> None:
        job = JobInput(
            idea="A concise product reveal with one person opening a small case.",
            duration_sec=6,
            use_voice=False,
            orientation="landscape",
            resolution="draft",
        )

        plan = self.planner.build_plan(job)
        scene = plan.scenes[0]
        prompt = scene.prompt_text
        metadata = scene.prompt_build_metadata

        self.assertEqual(metadata["builder_version"], "phaseA_scene_world_contract_v2")
        self.assertIn("scene_world_contract", metadata)
        self.assertIn("WORLD / SETTING:", prompt)
        self.assertIn("SUBJECT / ACTION:", prompt)
        self.assertIn("CAMERA / LIGHTING:", prompt)
        self.assertIn("STYLE LOCK:", prompt)
        self.assertIn("ALLOWED VISUALS:", prompt)
        self.assertIn("FORBIDDEN VISUALS:", prompt)
        self.assertIn("TEXT RISK POLICY:", prompt)
        self.assertIn("readable text", prompt.lower())
        self.assertIn("screens or ui facing camera", prompt.lower())

    def test_social_tip_scene_and_variation_prompts_keep_hard_forbidden_cues(self) -> None:
        job = JobInput(
            idea="Vier kurze Focus-Break Moves fuer mehr Klarheit im Arbeitstag.",
            script=(
                "Wenn dein Kopf am Bildschirm dicht macht, steh kurz auf. "
                "Trink erst einen Schluck Wasser, roll die Schultern und atme am Fenster tief durch. "
                "Dann geh mit ruhigerem Blick zurueck an den Tisch."
            ),
            duration_sec=18,
            use_voice=True,
            use_storyboard=True,
            use_music=True,
            orientation="portrait",
            resolution="draft",
            metadata={"subtitle_mode": "burn", "variations_per_scene": 4},
        )

        plan = self.planner.build_plan(job)
        scene = plan.scenes[0]
        scene_prompt = scene.prompt_text.lower()
        variation_prompt = scene.variations[0].prompt_variant_text.lower()

        self.assertIn("social format contract:", scene_prompt)
        for cue in [
            "no readable text",
            "no handwriting",
            "no document surfaces",
            "no screens or ui toward camera",
            "no labels",
            "typography",
            "glyphs",
            "letters",
            "numbers",
        ]:
            self.assertIn(cue, scene_prompt)
        forbidden_props = [
            value.lower() for value in scene.prompt_build_metadata["scene_world_contract"]["forbidden_props"]
        ]
        for cue in ["paper", "notebook", "screens", "visible ui", "labels", "office desk paper drift"]:
            self.assertIn(cue, forbidden_props)

        self.assertIn("world contract remains active", variation_prompt)
        self.assertIn("forbidden visuals still apply", variation_prompt)
        self.assertIn("do not introduce new props", variation_prompt)
        self.assertIn("paper", variation_prompt)
        self.assertIn("screens", variation_prompt)
        self.assertIn("readable text", variation_prompt)
        self.assertTrue(scene.variations[0].prompt_build_metadata["contract_preserved"])
        tactile_variations = [
            variation
            for scene in plan.scenes
            for variation in scene.variations
            if variation.shot_type == "detail_closeup"
        ]
        self.assertTrue(tactile_variations)
        for variation in tactile_variations:
            delta = (variation.prompt_delta or "").lower()
            self.assertNotIn("interfaces", delta)
            self.assertIn("without screens", delta)
            self.assertIn("paper", delta)

    def test_morning_reset_visual_prompt_sanitizer_removes_positive_device_ui_terms(self) -> None:
        job = JobInput(
            idea="Morning Reset social clip for TikTok content without phone distractions.",
            script=(
                "Atme kurz durch. Stell das Wasserglas ab. "
                "Oeffne das Fenster und starte klar in den Morgen."
            ),
            duration_sec=9,
            use_voice=True,
            use_storyboard=True,
            orientation="portrait",
            resolution="draft",
            metadata={"subtitle_mode": "off", "takes_per_scene": 2},
        )

        plan = self.planner.build_plan(job)
        forbidden_positive_terms = ["phone", "smartphone", "screen", "ui", "app", "website", "browser", "social media frame"]
        for scene in plan.scenes:
            contract = scene.prompt_build_metadata["scene_world_contract"]
            positive_text = " ".join(
                [
                    contract["visible_subject"],
                    contract["visual_anchor"],
                    contract["environment"],
                    contract["action"],
                    " ".join(contract["allowed_props"]),
                ]
            ).lower()
            for term in forbidden_positive_terms:
                self.assertIsNone(re.search(rf"\b{re.escape(term)}s?\b", positive_text))
            self.assertNotIn("readable human action", scene.prompt_text.lower())
            allowed_props = [value.lower() for value in contract["allowed_props"]]
            self.assertFalse(any(any(term in prop for term in forbidden_positive_terms) for prop in allowed_props))
            forbidden_props = " ".join(contract["forbidden_props"]).lower()
            for term in ["phones", "screens", "user interface", "app layout", "website", "social media frame"]:
                self.assertIn(term, forbidden_props)


if __name__ == "__main__":
    unittest.main()
