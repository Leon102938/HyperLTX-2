from __future__ import annotations

from pathlib import Path
from typing import Any

from .media_tooling import has_audio_stream


def native_a2vid_audio_policy(input_audio_path: str | Path, output_video_path: str | Path | None = None) -> dict[str, Any]:
    input_path = Path(input_audio_path)
    output_audio_present = has_audio_stream(output_video_path) if output_video_path and Path(output_video_path).exists() else None
    return {
        "native_audio_image_to_video": True,
        "input_audio_path": str(input_path),
        "input_audio_required": True,
        "input_audio_exists": input_path.exists(),
        "output_audio_present": output_audio_present,
        "audio_source_policy": "A2Vid must condition on the matching Qwen TTS chunk; arbitrary generated LTX voice is not accepted.",
        "audio_preserved_or_generated": "preserved_input_audio_expected",
        "needs_assembly_voice_mix": output_audio_present is not True,
    }


def fallback_audio_policy() -> dict[str, Any]:
    return {
        "native_audio_image_to_video": False,
        "fallback_mode": "image_to_video_plus_assembly_audio",
        "ltx_output_audio_policy": "remove_always",
        "master_audio": "qwen_tts_voice",
        "music_policy": "optional_under_voice_after_valid_riff_wav",
        "needs_assembly_voice_mix": True,
    }
