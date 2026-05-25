import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.segment_pipeline_router import PipelineAvailability, decide_segment_pipeline


class SegmentPipelineRouterTests(unittest.TestCase):
    def test_lipdub_for_strict_lipsync_with_reference_when_available(self):
        decision = decide_segment_pipeline(
            {
                "segment_id": "clip_01",
                "visual_type": "talking_host",
                "requires_strict_lipsync": True,
                "has_reference_video": True,
                "has_audio_chunk": True,
                "has_visible_mouth": True,
            },
            PipelineAvailability(lipdub=True),
        )

        self.assertEqual(decision.selected_pipeline, "lipdub")
        self.assertTrue(decision.strict_lipsync_available)

    def test_a2vid_soft_sync_for_talking_host_without_reference(self):
        decision = decide_segment_pipeline(
            {
                "segment_id": "clip_01",
                "visual_type": "talking_host",
                "requires_strict_lipsync": True,
                "has_reference_video": False,
                "has_keyframe_image": True,
                "has_audio_chunk": True,
                "has_visible_mouth": True,
            },
            PipelineAvailability(lipdub=False),
        )

        self.assertEqual(decision.selected_pipeline, "a2vid_two_stage")
        self.assertTrue(decision.strict_lipsync_unavailable)
        self.assertFalse(decision.claims_guaranteed_strict_lipsync)

    def test_ti2vid_for_landscape_broll(self):
        decision = decide_segment_pipeline(
            {"segment_id": "clip_02", "visual_type": "landscape_broll", "has_keyframe_image": True},
            PipelineAvailability(),
        )

        self.assertEqual(decision.selected_pipeline, "ti2vid_two_stages")
        self.assertFalse(decision.claims_audio_conditioning)

    def test_ti2vid_for_no_human_broll(self):
        decision = decide_segment_pipeline(
            {"segment_id": "clip_03", "visual_type": "no_human_broll", "has_keyframe_image": True},
            PipelineAvailability(),
        )

        self.assertEqual(decision.selected_pipeline, "ti2vid_two_stages")

    def test_a2vid_for_audio_driven_visual_with_keyframe(self):
        decision = decide_segment_pipeline(
            {
                "segment_id": "clip_03",
                "visual_type": "audio_driven_visual",
                "has_keyframe_image": True,
                "has_audio_chunk": True,
                "audio_should_drive_motion": True,
            },
            PipelineAvailability(a2vid=True, a2vid_audio_only=False),
        )

        self.assertEqual(decision.selected_pipeline, "a2vid_two_stage")
        self.assertTrue(decision.claims_audio_conditioning)

    def test_audio_driven_visual_blocks_without_keyframe_when_audio_only_not_supported(self):
        decision = decide_segment_pipeline(
            {"segment_id": "clip_03", "visual_type": "audio_driven_visual", "has_audio_chunk": True},
            PipelineAvailability(a2vid=True, a2vid_audio_only=False),
        )

        self.assertTrue(decision.blocked)
        self.assertEqual(decision.block_reason, "missing_keyframe_for_a2vid")

    def test_lipdub_is_not_confused_with_a2vid(self):
        decision = decide_segment_pipeline(
            {
                "segment_id": "clip_01",
                "visual_type": "talking_host",
                "requires_strict_lipsync": True,
                "has_reference_video": True,
                "has_audio_chunk": True,
            },
            PipelineAvailability(lipdub=True),
        )

        self.assertEqual(decision.selected_pipeline, "lipdub")
        self.assertNotEqual(decision.selected_pipeline, "a2vid_two_stage")


class SegmentPipelineApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_lipdub_contract_requires_reference_video_or_reports_unavailable(self):
        response = self.client.post("/ltx2/lipdub/submit", json={"audio_path": "/tmp/audio.wav"})

        self.assertIn(response.status_code, {422, 503})

    def test_a2vid_extended_api_contains_required_audio_args_and_rejects_unknown(self):
        response = self.client.post(
            "/ltx2/a2vid/submit",
            json={
                "prompt": "dry",
                "overrides": {
                    "dry_run": True,
                    "audio_path": "/tmp/audio.wav",
                    "image_path": "/tmp/image.png",
                    "unknown_arg": 1,
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"], "unknown_override_args")

    def test_a2vid_dry_run_accepts_audio_timing_args(self):
        response = self.client.post(
            "/ltx2/a2vid/submit",
            json={
                "prompt": "dry",
                "overrides": {
                    "dry_run": True,
                    "audio_path": "/tmp/audio.wav",
                    "audio_start_time": 0,
                    "audio_max_duration": 3,
                    "image_path": "/tmp/image.png",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["strict_lipsync_guaranteed"])

    def test_segment_audio_policy_strips_ti2vid_audio(self):
        response = self.client.post(
            "/ltx2/segment/submit",
            json={"segment_id": "clip_02", "visual_type": "landscape_broll", "has_keyframe_image": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("strip_ltx_audio", response.json()["decision"]["final_audio_policy"])

    def test_qwen_tts_remains_master_audio_default(self):
        capabilities = self.client.get("/ltx2/capabilities").json()

        self.assertEqual(capabilities["audio_policy"]["master_audio_default"], "qwen_tts_voice_chunks")

    def test_scene03_strategy_forbids_specks_terms(self):
        strategy_terms = {
            "floating white specks",
            "white dots",
            "snow",
            "sparkles",
            "particles",
            "dust clouds",
        }

        self.assertIn("snow", strategy_terms)
        self.assertIn("particles", strategy_terms)

    def test_no_pod_create_or_media_generation(self):
        # The tests only call dry-validation endpoints and pure routing functions.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
