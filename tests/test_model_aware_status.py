import json
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.model_status import BOOT_PROFILE_PATH, BOOT_PROFILE_STATUS_PATH, EFFECTIVE_TOOLS_CONFIG_PATH


class ModelAwareStatusTests(unittest.TestCase):
    def setUp(self):
        BOOT_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BOOT_PROFILE_PATH.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        BOOT_PROFILE_STATUS_PATH.write_text(
            json.dumps(
                {
                    "loaded": True,
                    "boot_profile_path": str(BOOT_PROFILE_PATH),
                    "effective_tools_config_path": str(EFFECTIVE_TOOLS_CONFIG_PATH),
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        EFFECTIVE_TOOLS_CONFIG_PATH.write_text(
            "\n".join(
                [
                    "JUPYTER=on",
                    "FASTAPI=on",
                    "INIT_SCRIPT=on",
                    "HiDream_O1_Dev=on",
                    "Qwen_TTS_Tokenizer=off",
                    "Qwen_TTS_Model=off",
                    "Qwen3_VL_Review=off",
                    "Vision_Review_Model=off",
                    "Ace_Step1_5=off",
                    "DW_LTX2=off",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.client = TestClient(app)

    def tearDown(self):
        for path in (BOOT_PROFILE_PATH, BOOT_PROFILE_STATUS_PATH, EFFECTIVE_TOOLS_CONFIG_PATH):
            path.unlink(missing_ok=True)

    def test_models_status_returns_clean_json(self):
        response = self.client.get("/DW/models/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["boot_profile_loaded"])
        self.assertEqual(payload["boot_profile_path"], str(BOOT_PROFILE_PATH))
        self.assertEqual(payload["effective_tools_config_path"], str(EFFECTIVE_TOOLS_CONFIG_PATH))
        self.assertTrue(payload["models"]["hidream_image_o1"]["enabled"])
        self.assertTrue(payload["models"]["hidream_image_o1"]["required"])
        self.assertFalse(payload["models"]["ltx_video"]["enabled"])
        self.assertTrue(payload["models"]["ltx_video"]["disabled"])
        self.assertEqual(payload["models"]["ltx_video"]["message"], "disabled by boot profile")

    def test_model_ready_endpoint_returns_single_model(self):
        response = self.client.get("/DW/models/hidream_image_o1/ready")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_id"], "hidream_image_o1")
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["required"])

    def test_dw_ready_stays_compatible(self):
        response = self.client.get("/DW/ready")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("ready", payload)
        self.assertIn("message", payload)


if __name__ == "__main__":
    unittest.main()
