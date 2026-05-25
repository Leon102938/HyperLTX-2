from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


VisualType = Literal[
    "talking_host",
    "host_gesture",
    "landscape_broll",
    "no_human_broll",
    "audio_driven_visual",
    "repair_existing_clip",
    "transition_between_keyframes",
]


@dataclass
class SegmentPipelineRequest:
    segment_id: str
    visual_type: VisualType
    requires_strict_lipsync: bool = False
    requires_soft_sync: bool = False
    has_reference_video: bool = False
    has_keyframe_image: bool = False
    has_audio_chunk: bool = False
    has_visible_mouth: bool = False
    audio_should_drive_motion: bool = False
    final_audio_source: str = "qwen_tts_master"
    duration_sec: float | None = None
    quality_priority: str = "balanced"
    allowed_fallbacks: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SegmentPipelineRequest":
        required = {"segment_id", "visual_type"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing required segment fields: {', '.join(missing)}")
        return cls(
            segment_id=str(payload["segment_id"]),
            visual_type=payload["visual_type"],
            requires_strict_lipsync=bool(payload.get("requires_strict_lipsync", False)),
            requires_soft_sync=bool(payload.get("requires_soft_sync", False)),
            has_reference_video=bool(payload.get("has_reference_video", False)),
            has_keyframe_image=bool(payload.get("has_keyframe_image", False)),
            has_audio_chunk=bool(payload.get("has_audio_chunk", False)),
            has_visible_mouth=bool(payload.get("has_visible_mouth", False)),
            audio_should_drive_motion=bool(payload.get("audio_should_drive_motion", False)),
            final_audio_source=str(payload.get("final_audio_source", "qwen_tts_master")),
            duration_sec=float(payload["duration_sec"]) if payload.get("duration_sec") is not None else None,
            quality_priority=str(payload.get("quality_priority", "balanced")),
            allowed_fallbacks=list(payload.get("allowed_fallbacks", [])),
        )


@dataclass
class PipelineAvailability:
    a2vid: bool = True
    a2vid_audio_only: bool = False
    ti2vid: bool = True
    lipdub: bool = False
    retake: bool = True
    keyframe_interpolation: bool = False

    @classmethod
    def detect(cls) -> "PipelineAvailability":
        root = Path("/workspace/LTX-2/packages/ltx-pipelines/src/ltx_pipelines")
        return cls(
            a2vid=(root / "a2vid_two_stage.py").is_file(),
            a2vid_audio_only=False,
            ti2vid=(root / "ti2vid_two_stages.py").is_file(),
            lipdub=(root / "lipdub.py").is_file(),
            retake=(root / "retake.py").is_file(),
            keyframe_interpolation=(root / "keyframe_interpolation.py").is_file(),
        )


@dataclass
class SegmentPipelineDecision:
    segment_id: str
    selected_pipeline: str | None
    reason: str
    strict_lipsync_available: bool = False
    strict_lipsync_unavailable: bool = False
    requires_reference_video: bool = False
    requires_audio_path: bool = False
    requires_image_path: bool = False
    final_audio_policy: str = "qwen_tts_master"
    allowed_fallback: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    claims_audio_conditioning: bool = False
    claims_guaranteed_strict_lipsync: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audio_policy_for(pipeline: str | None) -> str:
    if pipeline == "ti2vid_two_stages":
        return "strip_ltx_audio_use_qwen_tts_master_in_assembly"
    if pipeline == "a2vid_two_stage":
        return "condition_on_audio_chunk_probe_output_keep_qwen_tts_master_by_default"
    if pipeline == "lipdub":
        return "probe_lipdub_output_audio_document_source"
    return "qwen_tts_master"


def decide_segment_pipeline(
    request: SegmentPipelineRequest | dict[str, Any],
    availability: PipelineAvailability | None = None,
) -> SegmentPipelineDecision:
    req = SegmentPipelineRequest.from_dict(request) if isinstance(request, dict) else request
    av = availability or PipelineAvailability.detect()

    if req.visual_type == "talking_host" and req.requires_strict_lipsync:
        if req.has_reference_video and av.lipdub:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="lipdub",
                reason="Strict lip-sync requested and LipDub plus reference video are available.",
                strict_lipsync_available=True,
                requires_reference_video=True,
                requires_audio_path=True,
                final_audio_policy=_audio_policy_for("lipdub"),
                claims_audio_conditioning=True,
                claims_guaranteed_strict_lipsync=True,
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline="a2vid_two_stage",
            reason="Strict lip-sync requested, but LipDub/reference video is unavailable; using A2Vid soft-sync fallback, not guaranteed strict lip-sync.",
            strict_lipsync_unavailable=True,
            requires_audio_path=True,
            requires_image_path=True,
            final_audio_policy=_audio_policy_for("a2vid_two_stage"),
            allowed_fallback="a2vid_soft_sync",
            claims_audio_conditioning=True,
            claims_guaranteed_strict_lipsync=False,
        )

    if req.visual_type in {"host_gesture", "talking_host"} and (req.requires_soft_sync or req.has_audio_chunk):
        if req.has_keyframe_image and req.has_audio_chunk and av.a2vid:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="a2vid_two_stage",
                reason="Talking/host segment has keyframe and audio chunk; A2Vid provides audio-driven soft-sync motion.",
                requires_audio_path=True,
                requires_image_path=True,
                final_audio_policy=_audio_policy_for("a2vid_two_stage"),
                claims_audio_conditioning=True,
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline=None,
            reason="Host soft-sync requires both keyframe image and audio chunk.",
            requires_audio_path=True,
            requires_image_path=True,
            blocked=True,
            block_reason="missing_keyframe_or_audio_for_a2vid",
        )

    if req.visual_type in {"landscape_broll", "no_human_broll"}:
        if req.has_keyframe_image and av.ti2vid:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="ti2vid_two_stages",
                reason="B-roll/no-human segment is image/prompt driven; audio is handled later in assembly.",
                requires_image_path=True,
                final_audio_policy=_audio_policy_for("ti2vid_two_stages"),
                claims_audio_conditioning=False,
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline=None,
            reason="B-roll route requires a keyframe image and TI2Vid availability.",
            requires_image_path=True,
            blocked=True,
            block_reason="missing_keyframe_for_ti2vid",
        )

    if req.visual_type == "audio_driven_visual":
        if not req.has_audio_chunk:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline=None,
                reason="Audio-driven visual requires an audio chunk.",
                requires_audio_path=True,
                blocked=True,
                block_reason="missing_audio_for_audio_driven_visual",
            )
        if av.a2vid_audio_only:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="a2vid_audio_only",
                reason="Audio-driven visual can use audio-only A2Vid on this backend.",
                requires_audio_path=True,
                final_audio_policy=_audio_policy_for("a2vid_two_stage"),
                claims_audio_conditioning=True,
            )
        if req.has_keyframe_image and av.a2vid:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="a2vid_two_stage",
                reason="Audio-driven visual uses A2Vid with keyframe image and audio chunk.",
                requires_audio_path=True,
                requires_image_path=True,
                final_audio_policy=_audio_policy_for("a2vid_two_stage"),
                claims_audio_conditioning=True,
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline=None,
            reason="A2Vid audio-only is unavailable and no keyframe image was provided.",
            requires_audio_path=True,
            requires_image_path=True,
            blocked=True,
            block_reason="missing_keyframe_for_a2vid",
        )

    if req.visual_type == "repair_existing_clip":
        if av.retake:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="retake",
                reason="Repair segment uses Retake when available.",
                final_audio_policy="preserve_or_rebuild_audio_in_assembly_after_probe",
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline=None,
            reason="Retake is unavailable; targeted retry is required.",
            allowed_fallback="targeted_retry",
            blocked=True,
            block_reason="retake_unavailable",
        )

    if req.visual_type == "transition_between_keyframes":
        if av.keyframe_interpolation:
            return SegmentPipelineDecision(
                segment_id=req.segment_id,
                selected_pipeline="keyframe_interpolation",
                reason="Keyframe interpolation is available for transition segment.",
                requires_image_path=True,
                final_audio_policy="assembly_audio_only",
            )
        return SegmentPipelineDecision(
            segment_id=req.segment_id,
            selected_pipeline="assembly_cut",
            reason="Keyframe interpolation is unavailable; use assembly cut.",
            allowed_fallback="assembly_cut",
            final_audio_policy="assembly_audio_only",
        )

    return SegmentPipelineDecision(
        segment_id=req.segment_id,
        selected_pipeline=None,
        reason=f"Unsupported visual_type: {req.visual_type}",
        blocked=True,
        block_reason="unsupported_visual_type",
    )


def capability_matrix() -> dict[str, Any]:
    av = PipelineAvailability.detect()
    return {
        "ti2vid_two_stages": {
            "available": av.ti2vid,
            "purpose": "image + prompt -> video",
            "native_audio_conditioning": False,
            "best_for": ["landscape_broll", "no_human_broll", "establishing_shots"],
            "audio_policy": "strip_output_audio_if_present; add master audio in assembly",
        },
        "a2vid_two_stage": {
            "available": av.a2vid,
            "purpose": "image + audio + prompt -> audio-driven video",
            "native_audio_conditioning": True,
            "sync_claim": "soft-sync/audio-driven motion; not guaranteed strict lip-sync",
            "required_inputs": ["image_path", "audio_path", "prompt"],
            "important_args": ["audio_path", "audio_start_time", "audio_max_duration", "image_path", "num_frames", "frame_rate"],
        },
        "lipdub": {
            "available": av.lipdub,
            "purpose": "reference video + audio -> lip dubbing/rephrasing",
            "requires_reference_video": True,
            "strict_lipsync_candidate": av.lipdub,
            "missing_components": [] if av.lipdub else ["ltx_pipelines.lipdub module"],
        },
        "retake": {
            "available": av.retake,
            "purpose": "repair/regenerate an existing clip region",
            "standard_main_pipeline": False,
        },
        "keyframe_interpolation": {
            "available": av.keyframe_interpolation,
            "purpose": "transitions between keyframes",
            "fallback": "assembly_cut",
        },
    }
