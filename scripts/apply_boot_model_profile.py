#!/usr/bin/env python3
from __future__ import annotations

import base64
import binascii
import json
import os
import re
from pathlib import Path
from typing import Any


BOOT_PROFILE_ENV = "CONTENT_MACHINE_BOOT_PROFILE_B64"
BOOT_PROFILE_JOB_ID_ENV = "CONTENT_MACHINE_BOOT_PROFILE_JOB_ID"
BOOT_PROFILE_SOURCE_ENV = "CONTENT_MACHINE_BOOT_PROFILE_SOURCE"

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
RUNTIME_DIR = Path(os.environ.get("BOOT_PROFILE_RUNTIME_DIR", str(WORKSPACE / "runtime")))
TOOLS_CONFIG_PATH = Path(os.environ.get("TOOLS_CONFIG_PATH", str(WORKSPACE / "tools.config")))
BOOT_PROFILE_PATH = Path(os.environ.get("BOOT_PROFILE_PATH", str(RUNTIME_DIR / "boot_model_profile.json")))
EFFECTIVE_TOOLS_CONFIG_PATH = Path(
    os.environ.get("EFFECTIVE_TOOLS_CONFIG_PATH", str(RUNTIME_DIR / "effective_tools.config"))
)
TOOLS_CONFIG_ORIGINAL_PATH = Path(
    os.environ.get("TOOLS_CONFIG_ORIGINAL_PATH", str(RUNTIME_DIR / "tools.config.original"))
)
STATUS_PATH = Path(os.environ.get("BOOT_PROFILE_STATUS_PATH", str(RUNTIME_DIR / "boot_model_profile_status.json")))

MODEL_TO_TOOLS = {
    "hidream_image_o1": ["HiDream_O1_Dev"],
    "ltx_video": ["DW_LTX2"],
    "ace_music": ["Ace_Step1_5"],
    "qwen_tts": ["Qwen_TTS_Tokenizer", "Qwen_TTS_Model"],
    "qwen3_vl_review": ["Qwen3_VL_Review", "Vision_Review_Model"],
}

PROFILE_CONTROLLED_TOOLS = sorted({tool for tools in MODEL_TO_TOOLS.values() for tool in tools})
VALID_SCHEMA_VERSION = "content_machine_boot_profile_v1"
ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_list(profile: dict[str, Any], key: str) -> list[str]:
    value = profile.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return value


def decode_profile_from_env(env: dict[str, str] | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    env = env or os.environ
    encoded = env.get(BOOT_PROFILE_ENV)
    if not encoded:
        return None, []

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{BOOT_PROFILE_ENV} is not valid base64: {exc}") from exc

    try:
        profile = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{BOOT_PROFILE_ENV} does not contain valid UTF-8 JSON: {exc}") from exc

    if not isinstance(profile, dict):
        raise ValueError("boot profile must be a JSON object")

    warnings = validate_profile(profile)
    if env.get(BOOT_PROFILE_JOB_ID_ENV) and not profile.get("job_id"):
        profile["job_id"] = env[BOOT_PROFILE_JOB_ID_ENV]
    if env.get(BOOT_PROFILE_SOURCE_ENV) and not profile.get("source"):
        profile["source"] = env[BOOT_PROFILE_SOURCE_ENV]
    return profile, warnings


def validate_profile(profile: dict[str, Any]) -> list[str]:
    schema_version = profile.get("schema_version")
    if schema_version != VALID_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")

    warnings: list[str] = []
    for key in (
        "required_models",
        "disabled_models",
        "readiness_wait_for",
        "tools_config_enable",
        "tools_config_disable",
    ):
        _string_list(profile, key)

    for key in ("required_models", "disabled_models", "readiness_wait_for"):
        for model_id in _string_list(profile, key):
            if model_id not in MODEL_TO_TOOLS:
                warnings.append(f"unknown model id in {key}: {model_id}")

    known_tools = set(PROFILE_CONTROLLED_TOOLS)
    for key in ("tools_config_enable", "tools_config_disable"):
        for tool in _string_list(profile, key):
            if tool not in known_tools:
                warnings.append(f"unknown tool in {key}: {tool}")

    return warnings


def _desired_tool_state(profile: dict[str, Any]) -> dict[str, str]:
    state = {tool: "off" for tool in PROFILE_CONTROLLED_TOOLS}

    for model_id in _string_list(profile, "required_models"):
        for tool in MODEL_TO_TOOLS.get(model_id, []):
            state[tool] = "on"

    for tool in _string_list(profile, "tools_config_enable"):
        if tool in state:
            state[tool] = "on"

    for model_id in _string_list(profile, "disabled_models"):
        for tool in MODEL_TO_TOOLS.get(model_id, []):
            state[tool] = "off"

    for tool in _string_list(profile, "tools_config_disable"):
        if tool in state:
            state[tool] = "off"

    return state


def build_effective_tools_config(default_config: str, profile: dict[str, Any]) -> str:
    state = _desired_tool_state(profile)
    seen: set[str] = set()
    output: list[str] = []

    for line in default_config.splitlines():
        match = ASSIGNMENT_RE.match(line.strip())
        if match and match.group(1) in state:
            key = match.group(1)
            output.append(f"{key}={state[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key in PROFILE_CONTROLLED_TOOLS:
        if key not in seen:
            output.append(f"{key}={state[key]}")

    return "\n".join(output).rstrip() + "\n"


def apply_boot_model_profile(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env or os.environ
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    try:
        profile, warnings = decode_profile_from_env(env)
    except ValueError as exc:
        EFFECTIVE_TOOLS_CONFIG_PATH.unlink(missing_ok=True)
        payload = {
            "loaded": False,
            "error": str(exc),
            "boot_profile_path": str(BOOT_PROFILE_PATH),
            "effective_tools_config_path": str(EFFECTIVE_TOOLS_CONFIG_PATH),
            "default_behavior": True,
        }
        _write_json(STATUS_PATH, payload)
        return payload

    if profile is None:
        EFFECTIVE_TOOLS_CONFIG_PATH.unlink(missing_ok=True)
        payload = {
            "loaded": False,
            "message": f"{BOOT_PROFILE_ENV} not set; using default tools.config",
            "boot_profile_path": str(BOOT_PROFILE_PATH),
            "effective_tools_config_path": str(EFFECTIVE_TOOLS_CONFIG_PATH),
            "default_behavior": True,
        }
        _write_json(STATUS_PATH, payload)
        return payload

    default_config = TOOLS_CONFIG_PATH.read_text(encoding="utf-8") if TOOLS_CONFIG_PATH.exists() else ""
    _write_json(BOOT_PROFILE_PATH, profile)
    TOOLS_CONFIG_ORIGINAL_PATH.write_text(default_config, encoding="utf-8")
    effective_config = build_effective_tools_config(default_config, profile)
    EFFECTIVE_TOOLS_CONFIG_PATH.write_text(effective_config, encoding="utf-8")

    payload = {
        "loaded": True,
        "warnings": warnings,
        "job_id": profile.get("job_id"),
        "source": profile.get("source"),
        "boot_profile_path": str(BOOT_PROFILE_PATH),
        "effective_tools_config_path": str(EFFECTIVE_TOOLS_CONFIG_PATH),
        "default_behavior": False,
    }
    _write_json(STATUS_PATH, payload)
    return payload


def main() -> int:
    result = apply_boot_model_profile()
    if result.get("loaded"):
        print(f"[boot-profile] loaded: {result['boot_profile_path']}")
        print(f"[boot-profile] effective tools config: {result['effective_tools_config_path']}")
        for warning in result.get("warnings", []):
            print(f"[boot-profile] warning: {warning}")
    else:
        print(f"[boot-profile] default behavior: {result.get('message') or result.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
