from __future__ import annotations

from dataclasses import dataclass

from agent_core.creative_os.cockpit.state_adapter import CockpitState


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    title: str
    short_description: str
    artifacts: tuple[str, ...]
    next_action: str
    view_type: str = "placeholder"


STAGE_DEFINITIONS: tuple[StageDefinition, ...] = (
    StageDefinition("00", "Command Center", "Run vorbereiten · Parameter prüfen · Startstatus kontrollieren", tuple(), "Startparameter prüfen", "command_center"),
    StageDefinition("01", "Pipeline wählen", "Operator-Pipeline anzeigen und spaetere Auswahl vorbereiten.", ("plan.json", "intent_route.json"), "Pipeline selection is preview-only in V0.1", "pipeline_select"),
    StageDefinition("02", "Mode & Style", "Mode, Style, Format und Topic fuer den Run sichtbar machen.", ("normalized_job.json", "plan.json"), "Review mode/style inputs before later command composition", "mode_style"),
    StageDefinition("03", "Skills laden", "Skill Health, geladene Skills, Fallbacks und fehlende Skills zeigen.", ("skill_match.json",), "Inspect skill coverage; no runtime loading is triggered", "skills"),
    StageDefinition("04", "Creative Strategy", "Creative Strategy, Director-Plan und Risiken zusammenfassen.", ("creative_strategy.json", "plan.json", "director_output.json"), "Detailed strategy panel planned"),
    StageDefinition("05", "Beat / Hook Planner", "Hook, Beats, Escalation und Payoff vorbereiten.", ("selected_beat_plan.json",), "Detailed beat panel planned"),
    StageDefinition("06", "Creative Judge", "Judge-Ergebnis, Auswahl und Rationale anzeigen.", ("creative_judge.json", "decision_log.json"), "Detailed judge panel planned"),
    StageDefinition("07", "Scene Contracts", "Scene Contracts mit Umgebung, Action, Kamera und Controls zeigen.", ("scene_contracts.json",), "Detailed scene contract panel planned"),
    StageDefinition("08", "Image Prompt Compiler", "Bildprompt-Listen und Prompt-Audit vorbereiten.", ("zimage_prompts.json", "model_prompts.json", "prompt_audit.json"), "Detailed image prompt panel planned"),
    StageDefinition("09", "Image / Keyframe Generation", "Keyframe- und Image-Job-Status im bestehenden Panel zeigen.", ("keyframe_manifest.json", "keyframes/scene_01.png", "keyframes/scene_02.png", "keyframes/scene_03.png"), "Inspect generated keyframes", "image_jobs"),
    StageDefinition("10", "Keyframe Review", "Keyframe-QA und Review-Entscheidungen anzeigen.", ("keyframe_review.json", "stage6_review_decision.json"), "Detailed keyframe review panel planned"),
    StageDefinition("11", "LTX Motion Prompt Compiler", "LTX-I2V-Motion-Prompts und Audit anzeigen.", ("ltx_motion_prompts.json", "ltx_prompt_audit.json"), "Detailed LTX prompt panel planned"),
    StageDefinition("12", "LTX Video Generation", "Video-Take-Manifest und spaetere Video-Dateien zeigen.", ("ltx_video_takes_manifest.json",), "Render remains disabled; detailed video generation panel planned"),
    StageDefinition("13", "Video Review", "Video Review und Quality Findings anzeigen.", ("video_review.json",), "Detailed video review panel planned"),
    StageDefinition("14", "Final Assembly", "Final MP4, Untertitel, Voice, Music und Overlay zusammenfassen.", ("final.mp4", "subtitles.srt", "voice.wav", "music.wav"), "Detailed assembly panel planned"),
    StageDefinition("15", "Final Output", "Final Verdict, Download-/Statusdaten und finale Artefakte zeigen.", ("final_quality_verdict.json", "final.mp4"), "Detailed final output panel planned"),
)

STAGE_BY_ID: dict[str, StageDefinition] = {stage.stage_id: stage for stage in STAGE_DEFINITIONS}
DEFAULT_STAGE_ID = "09"


def stage_ids() -> tuple[str, ...]:
    return tuple(stage.stage_id for stage in STAGE_DEFINITIONS)


def normalize_stage_id(stage_id: str | None) -> str:
    if stage_id in STAGE_BY_ID:
        return str(stage_id)
    return DEFAULT_STAGE_ID


def selected_stage_id(state: CockpitState) -> str:
    return normalize_stage_id(getattr(state, "selected_stage", DEFAULT_STAGE_ID))


def selected_stage(state: CockpitState) -> StageDefinition:
    return STAGE_BY_ID[selected_stage_id(state)]


def status_for_stage(state: CockpitState, stage: StageDefinition) -> str:
    current = _current_stage_id(state)
    if stage.stage_id == current:
        return "current"
    if stage.stage_id in {"00", "01", "02"} and state.run_found:
        return "passed"
    if stage.stage_id == "03" and state.skill_health.status in {"ok", "warning"}:
        return "passed"
    if stage.stage_id == "04" and _any_artifact_present(state, stage.artifacts):
        return "passed"
    if stage.stage_id in {"05", "06", "07", "08", "09", "10", "11"} and _any_artifact_present(state, stage.artifacts):
        return "passed"
    if stage.stage_id in {"14", "15"} and _artifact_present(state, "final.mp4"):
        return "passed"
    if _any_artifact_present(state, stage.artifacts):
        return "passed"
    return "pending"


def current_stage_id(state: CockpitState) -> str:
    return _current_stage_id(state)


def _current_stage_id(state: CockpitState) -> str:
    status = state.inspection.status
    if status.startswith("phase1_"):
        return "09"
    if status in {"ready_for_ltx_i2v_takes", "ready_for_stage_8"}:
        return "12"
    if _artifact_present(state, "ltx_motion_prompts.json"):
        return "12"
    if _artifact_present(state, "keyframe_review.json"):
        return "11"
    if _artifact_present(state, "keyframe_manifest.json"):
        return "10"
    if _artifact_present(state, "zimage_prompts.json") or _artifact_present(state, "model_prompts.json"):
        return "09"
    if _artifact_present(state, "scene_contracts.json") or _artifact_present(state, "stage_contracts.json"):
        return "08"
    if _artifact_present(state, "selected_beat_plan.json"):
        return "06"
    if _artifact_present(state, "creative_strategy.json"):
        return "05"
    if state.run_found:
        return "03"
    return DEFAULT_STAGE_ID


def artifact_status(state: CockpitState, artifact: str) -> str:
    return "present" if _artifact_present(state, artifact) else "missing"


def _any_artifact_present(state: CockpitState, artifacts: tuple[str, ...]) -> bool:
    return any(_artifact_present(state, artifact) for artifact in artifacts)


def _artifact_present(state: CockpitState, artifact: str) -> bool:
    if state.inspection.artifacts.get(artifact):
        return True
    return (state.data_source_path / artifact).exists()
