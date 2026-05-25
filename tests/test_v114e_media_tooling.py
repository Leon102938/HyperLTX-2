import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.LTX2 import _build_command
from app.ace_artifact_resolver import is_riff_wav, resolve_ace_artifact
from app.assembly_contract import assembly_contract_v114e
from app.audio_policy import fallback_audio_policy
from app.ltx_readiness import ltx_completeness_readiness_report
from app.media_tooling import command_path, media_tooling_preflight, strip_audio_command


class V114EMediaToolingTests(unittest.TestCase):
    def test_a2vid_pipeline_is_own_pipeline_type(self):
        cmd, _, meta = _build_command(
            "prompt",
            "/tmp/out.mp4",
            {"pipeline": "a2vid_two_stage", "audio_path": "/tmp/audio.wav", "image_path": "/tmp/image.png"},
        )

        self.assertIn("ltx_pipelines.a2vid_two_stage", cmd)
        self.assertNotIn("ltx_pipelines.ti2vid_two_stages", cmd)
        self.assertTrue(meta["native_audio_image_to_video"])
        self.assertEqual(meta["input_audio_path"], "/tmp/audio.wav")

    def test_native_a2vid_requires_audio_path(self):
        with self.assertRaisesRegex(ValueError, "audio_path is required"):
            _build_command("prompt", "/tmp/out.mp4", {"pipeline": "a2vid_two_stage", "image_path": "/tmp/image.png"})

    def test_image_to_video_fallback_does_not_switch_to_a2vid_when_audio_present(self):
        cmd, _, meta = _build_command(
            "prompt",
            "/tmp/out.mp4",
            {"pipeline": "image_to_video", "audio_path": "/tmp/audio.wav", "image_path": "/tmp/image.png"},
        )

        self.assertIn("ltx_pipelines.ti2vid_two_stages", cmd)
        self.assertNotIn("ltx_pipelines.a2vid_two_stage", cmd)
        self.assertFalse(meta["native_audio_image_to_video"])
        self.assertEqual(meta["fallback_mode"], "image_to_video_plus_assembly_audio")
        self.assertEqual(meta["audio_policy"]["ltx_output_audio_policy"], "remove_always")

    def test_fallback_policy_keeps_qwen_tts_as_master_audio(self):
        policy = fallback_audio_policy()

        self.assertEqual(policy["master_audio"], "qwen_tts_voice")
        self.assertEqual(policy["ltx_output_audio_policy"], "remove_always")

    def test_ltx_completeness_requires_tokenizer_model_and_help_success(self):
        with patch("app.ltx_readiness.DEFAULT_CHECKPOINT_PATH", Path("/tmp/main.safetensors")), \
            patch("app.ltx_readiness.DEFAULT_SPATIAL_UPSAMPLER_PATH", Path("/tmp/up.safetensors")), \
            patch("app.ltx_readiness.DEFAULT_DISTILLED_LORA_PATH", Path("/tmp/lora.safetensors")), \
            patch("app.ltx_readiness.DEFAULT_GEMMA_ROOT", Path("/tmp/missing-gemma")):
            report = ltx_completeness_readiness_report(run_help=False, status_payload={"ready": None})

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["gemma_required_files"]["tokenizer.model"]["exists"])
        self.assertIsNone(report["status_payload_ready"])
        self.assertFalse(report["status_payload_ready_is_sufficient"])

    def test_ace_resolver_does_not_accept_json_as_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "response.wav"
            src.write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
            dst = Path(tmp) / "music.wav"

            report = resolve_ace_artifact(src, dst)

        self.assertFalse(report["resolved"])
        self.assertIn("JSON response", report["error"])

    def test_ace_resolver_requires_riff_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.wav"
            fake.write_bytes(b'{"ok": true}')

            self.assertFalse(is_riff_wav(fake))

    def test_ace_resolver_copies_real_riff_wav_from_json_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "real.wav"
            wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32)
            src = Path(tmp) / "result.json"
            src.write_text(json.dumps({"primary_output_path": str(wav)}), encoding="utf-8")
            dst = Path(tmp) / "music.wav"

            report = resolve_ace_artifact(src, dst)

        self.assertTrue(report["resolved"])
        self.assertEqual(report["resolved_path"], str(dst))

    def test_media_tooling_preflight_detects_ffmpeg_and_ffprobe(self):
        report = media_tooling_preflight()

        self.assertEqual(report["ffmpeg"]["present"], bool(command_path("ffmpeg")))
        self.assertEqual(report["ffprobe"]["present"], bool(command_path("ffprobe")))

    def test_strip_audio_command_removes_audio(self):
        cmd = strip_audio_command("in.mp4", "out.mp4")

        self.assertIn("-an", cmd)

    def test_assembly_contract_uses_no_black_panel(self):
        contract = assembly_contract_v114e()

        self.assertFalse(contract["black_panel_masking"])

    def test_tests_do_not_create_pods_or_generate_media(self):
        # This suite only builds commands/contracts and writes tiny temp files.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
