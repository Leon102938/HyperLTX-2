import base64
import json
import os
import unittest
from pathlib import Path

from scripts.apply_boot_model_profile import (
    BOOT_PROFILE_ENV,
    BOOT_PROFILE_PATH,
    EFFECTIVE_TOOLS_CONFIG_PATH,
    STATUS_PATH,
    apply_boot_model_profile,
    build_effective_tools_config,
)


DEFAULT_CONFIG = """JUPYTER=on
FASTAPI=on
INIT_SCRIPT=on
HiDream_O1_Dev=on
Qwen_TTS_Tokenizer=on
Qwen_TTS_Model=on
Qwen3_VL_Review=off
Vision_Review_Model=off
Ace_Step1_5=on
DW_LTX2=on
"""


def encoded_profile(**overrides):
    profile = {
        "schema_version": "content_machine_boot_profile_v1",
        "job_id": "unit-test",
        "pipeline": "image_only",
        "required_models": ["hidream_image_o1"],
        "disabled_models": ["ltx_video", "ace_music", "qwen_tts", "qwen3_vl_review"],
        "readiness_wait_for": ["hidream_image_o1"],
        "tools_config_enable": ["HiDream_O1_Dev"],
        "tools_config_disable": [
            "DW_LTX2",
            "Ace_Step1_5",
            "Qwen_TTS_Tokenizer",
            "Qwen_TTS_Model",
            "Qwen3_VL_Review",
            "Vision_Review_Model",
        ],
        "source": "hermes_content_machine_v1_7",
    }
    profile.update(overrides)
    return base64.b64encode(json.dumps(profile).encode("utf-8")).decode("ascii")


class BootModelProfileTests(unittest.TestCase):
    def tearDown(self):
        for path in (BOOT_PROFILE_PATH, EFFECTIVE_TOOLS_CONFIG_PATH, STATUS_PATH):
            path.unlink(missing_ok=True)

    def test_missing_env_keeps_default_behavior(self):
        result = apply_boot_model_profile(env={})

        self.assertFalse(result["loaded"])
        self.assertTrue(result["default_behavior"])
        self.assertFalse(EFFECTIVE_TOOLS_CONFIG_PATH.exists())

    def test_image_only_profile_enables_only_hidream(self):
        result = apply_boot_model_profile(env={BOOT_PROFILE_ENV: encoded_profile()})

        self.assertTrue(result["loaded"])
        self.assertTrue(BOOT_PROFILE_PATH.exists())
        effective = EFFECTIVE_TOOLS_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("HiDream_O1_Dev=on", effective)
        self.assertIn("DW_LTX2=off", effective)
        self.assertIn("Ace_Step1_5=off", effective)
        self.assertIn("Qwen_TTS_Tokenizer=off", effective)
        self.assertIn("Qwen_TTS_Model=off", effective)

    def test_video_music_profile_enables_ltx_and_ace(self):
        effective = build_effective_tools_config(
            DEFAULT_CONFIG,
            {
                "schema_version": "content_machine_boot_profile_v1",
                "required_models": ["ltx_video", "ace_music"],
                "disabled_models": ["hidream_image_o1", "qwen_tts", "qwen3_vl_review"],
                "tools_config_enable": [],
                "tools_config_disable": [],
            },
        )

        self.assertIn("DW_LTX2=on", effective)
        self.assertIn("Ace_Step1_5=on", effective)
        self.assertIn("HiDream_O1_Dev=off", effective)

    def test_voice_profile_enables_qwen_tts(self):
        effective = build_effective_tools_config(
            DEFAULT_CONFIG,
            {
                "schema_version": "content_machine_boot_profile_v1",
                "required_models": ["qwen_tts"],
                "disabled_models": ["hidream_image_o1", "ltx_video", "ace_music", "qwen3_vl_review"],
                "tools_config_enable": [],
                "tools_config_disable": [],
            },
        )

        self.assertIn("Qwen_TTS_Tokenizer=on", effective)
        self.assertIn("Qwen_TTS_Model=on", effective)
        self.assertIn("DW_LTX2=off", effective)

    def test_invalid_base64_does_not_crash(self):
        result = apply_boot_model_profile(env={BOOT_PROFILE_ENV: "not-valid-@@@"})

        self.assertFalse(result["loaded"])
        self.assertTrue(result["default_behavior"])
        self.assertIn("not valid base64", result["error"])
        self.assertFalse(EFFECTIVE_TOOLS_CONFIG_PATH.exists())

    def test_invalid_json_does_not_crash(self):
        bad_json = base64.b64encode(b"{").decode("ascii")
        result = apply_boot_model_profile(env={BOOT_PROFILE_ENV: bad_json})

        self.assertFalse(result["loaded"])
        self.assertTrue(result["default_behavior"])
        self.assertIn("valid UTF-8 JSON", result["error"])
        self.assertFalse(EFFECTIVE_TOOLS_CONFIG_PATH.exists())


if __name__ == "__main__":
    unittest.main()
