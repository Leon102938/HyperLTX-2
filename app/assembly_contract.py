from __future__ import annotations

from typing import Any


def assembly_contract_v114e() -> dict[str, Any]:
    return {
        "version": "v114e",
        "clip_order": "voice_chunk_order",
        "normalize_video": {"width": 1080, "height": 1920, "fps": 24},
        "clip_audio": "remove_when_fallback_or_untrusted_source",
        "voice_audio": "qwen_tts_master",
        "music": {"enabled_if_valid_riff_wav": True, "duck_under_voice": True, "target_role": "quiet_background"},
        "subtitles": {"mode": "soft_subtitles_or_controlled_burn_in", "source": "srt"},
        "black_panel_masking": False,
        "final_container": "mp4",
        "preflight_required": ["ffmpeg", "ffprobe", "readable_wav", "readable_mp4", "concat_supported"],
    }
