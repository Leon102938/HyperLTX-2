from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


WORKSPACE = Path("/workspace")
RUNTIME_DIR = WORKSPACE / "runtime"
BOOT_PROFILE_PATH = RUNTIME_DIR / "boot_model_profile.json"
BOOT_PROFILE_STATUS_PATH = RUNTIME_DIR / "boot_model_profile_status.json"
EFFECTIVE_TOOLS_CONFIG_PATH = RUNTIME_DIR / "effective_tools.config"
DEFAULT_TOOLS_CONFIG_PATH = WORKSPACE / "tools.config"

STATUS_DIR = WORKSPACE / "status"
INIT_FLAG = STATUS_DIR / "init_done"
HIDREAM_FLAG = STATUS_DIR / "hidream_ready"
QWEN_ENV_FLAG = STATUS_DIR / "qwen_tts_env_ready"
QWEN_TOKENIZER_FLAG = STATUS_DIR / "qwen_tts_tokenizer_ready"
QWEN_MODEL_FLAG = STATUS_DIR / "qwen_tts_model_ready"
QWEN_RUNTIME_FLAG = STATUS_DIR / "qwen_tts_runtime_ready"
ACE_STEP_READY_FLAG = STATUS_DIR / "ace_step_ready"
ACE_STEP_ENV_FLAG = STATUS_DIR / "ace_step_env_ready"
QWEN3_VL_READY_FLAG = STATUS_DIR / "qwen3_vl_ready"

ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

MODEL_TO_TOOLS = {
    "hidream_image_o1": ["HiDream_O1_Dev"],
    "ltx_video": ["DW_LTX2"],
    "ace_music": ["Ace_Step1_5"],
    "qwen_tts": ["Qwen_TTS_Tokenizer", "Qwen_TTS_Model"],
    "qwen3_vl_review": ["Qwen3_VL_Review", "Vision_Review_Model"],
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_tools_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return config
    for line in lines:
        match = ASSIGNMENT_RE.match(line.strip())
        if match:
            config[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return config


def _active_tools_config_path() -> Path:
    if EFFECTIVE_TOOLS_CONFIG_PATH.exists():
        return EFFECTIVE_TOOLS_CONFIG_PATH
    return DEFAULT_TOOLS_CONFIG_PATH


def _tool_enabled(config: dict[str, str], tools: list[str]) -> bool:
    return any(config.get(tool, "off").lower() == "on" for tool in tools)


def _all_tools_enabled(config: dict[str, str], tools: list[str]) -> bool:
    return all(config.get(tool, "off").lower() == "on" for tool in tools)


def _flag_ready(path: Path) -> bool:
    return path.exists()


def _ready_payload(model_id: str) -> tuple[bool | None, bool | None, bool, str]:
    if model_id == "hidream_image_o1":
        ready = _flag_ready(HIDREAM_FLAG)
        return ready, False, False, "ready endpoint ok" if ready else "hidream ready flag missing"
    if model_id == "qwen_tts":
        ready = all(_flag_ready(path) for path in (QWEN_ENV_FLAG, QWEN_TOKENIZER_FLAG, QWEN_MODEL_FLAG, QWEN_RUNTIME_FLAG))
        loading = _flag_ready(QWEN_ENV_FLAG) and not ready
        return ready, loading, False, "ready endpoint ok" if ready else "Qwen TTS is not fully ready"
    if model_id == "ace_music":
        ready = _flag_ready(ACE_STEP_READY_FLAG) and _flag_ready(ACE_STEP_ENV_FLAG)
        loading = _flag_ready(ACE_STEP_ENV_FLAG) and not ready
        return ready, loading, False, "ready endpoint ok" if ready else "ACE-Step 1.5 is not fully ready"
    if model_id == "qwen3_vl_review":
        ready = _flag_ready(QWEN3_VL_READY_FLAG)
        return ready, False, False, "ready flag ok" if ready else "Qwen3-VL ready flag missing"
    if model_id == "ltx_video":
        downloaded = _flag_ready(INIT_FLAG) if _flag_ready(INIT_FLAG) else None
        return None, False, False, "no model-specific ready endpoint"
    return None, None, True, "unknown model id"


def get_model_status(model_id: str) -> dict[str, Any]:
    profile = _read_json(BOOT_PROFILE_PATH) or {}
    status = _read_json(BOOT_PROFILE_STATUS_PATH) or {}
    config_path = _active_tools_config_path()
    config = _read_tools_config(config_path)
    tools = MODEL_TO_TOOLS.get(model_id, [])

    required_models = profile.get("required_models", []) if isinstance(profile.get("required_models", []), list) else []
    disabled_models = profile.get("disabled_models", []) if isinstance(profile.get("disabled_models", []), list) else []
    boot_profile_loaded = bool(status.get("loaded")) and BOOT_PROFILE_PATH.exists()

    enabled = _all_tools_enabled(config, tools) if model_id == "qwen_tts" else _tool_enabled(config, tools)
    required = model_id in required_models
    disabled = model_id in disabled_models or (boot_profile_loaded and not enabled)
    ready, loading, failed, message = _ready_payload(model_id)
    downloaded = None

    if disabled:
        ready = False
        loading = False
        failed = False
        message = "disabled by boot profile" if model_id in disabled_models or boot_profile_loaded else "disabled by config"
    elif not enabled:
        ready = False if ready is not None else ready
        loading = False
        message = "disabled by config"

    return {
        "enabled": enabled,
        "required": required,
        "disabled": disabled,
        "downloaded": downloaded,
        "loading": loading,
        "ready": ready,
        "failed": failed,
        "message": message,
        "tools": tools,
    }


def get_models_status() -> dict[str, Any]:
    status = _read_json(BOOT_PROFILE_STATUS_PATH) or {}
    boot_profile_loaded = bool(status.get("loaded")) and BOOT_PROFILE_PATH.exists()
    return {
        "boot_profile_loaded": boot_profile_loaded,
        "boot_profile_path": str(BOOT_PROFILE_PATH),
        "boot_profile_status_path": str(BOOT_PROFILE_STATUS_PATH),
        "effective_tools_config_path": str(EFFECTIVE_TOOLS_CONFIG_PATH),
        "active_tools_config_path": str(_active_tools_config_path()),
        "profile_error": status.get("error"),
        "warnings": status.get("warnings", []),
        "models": {model_id: get_model_status(model_id) for model_id in MODEL_TO_TOOLS},
    }
