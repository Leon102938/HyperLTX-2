from __future__ import annotations

import json

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import rows_text
from agent_core.creative_os.cockpit.state_adapter import CockpitState
from agent_core.creative_os.cockpit.stage_registry import STAGE_DEFINITIONS, StageDefinition, artifact_status, selected_stage
from agent_core.creative_os.cockpit.theme import BG_WORKSPACE, TEXT_ACTIVE, TEXT_LABEL, TEXT_MAIN, TEXT_MUTED, TEXT_SUCCESS, style

BOX_WIDTH = 132
SCENE_CONTRACT_A_WIDTH = 30
SCENE_CONTRACT_B_WIDTH = 54
SCENE_CONTRACT_C_WIDTH = 44
JOB_PREVIEW_WIDTH = 18
JOB_MAIN_WIDTH = 78
JOB_STATUS_WIDTH = 20
JOB_BLOCK_WIDTH = 126
JOB_CARD_PREVIEW_WIDTH = 18
JOB_CARD_MAIN_WIDTH = 72
JOB_CARD_STATUS_WIDTH = 19
JOB_CARD_INNER_WIDTH = 122


def render(state: CockpitState) -> Text:
    if not state.run_found:
        return _missing_run_text(state)
    stage = selected_stage(state)
    if stage.view_type != "image_jobs":
        return _stage_workspace(state, stage)
    return _image_jobs_workspace(state)


def _image_jobs_workspace(state: CockpitState) -> Text:
    data = state.workspace
    workspace = Text()
    workspace.append(
        _section_box(
            "CURRENT POSITION",
            _position_grid(
                (
                    ("Current Step", data.current_step),
                    ("Operator Focus", data.operator_focus),
                    ("Render Paused", data.render_paused),
                    ("Final MP4", _final_mp4_status(state)),
                    ("Last Passed", data.last_passed),
                    ("Next Technical", data.next_technical),
                    ("Director Mode", _director_mode(state)),
                    ("Run Type", _run_type(state)),
                )
            ),
        )
    )
    workspace.append(_section_box("READINESS / INPUTS", _readiness_input_lines(state)))
    job_lines = _prompt_job_lines(state)
    workspace.append(_section_box("PROMPTS / IMAGE JOBS", job_lines))
    return workspace


def _readiness_input_lines(state: CockpitState) -> list[Text]:
    return [
        _readiness_line("PIPELINE / ROUTE", _pipeline_route_summary(state)),
        _readiness_line("CREATIVE INPUTS", _creative_inputs_summary(state)),
        _readiness_line("PROMPT INPUTS", _prompt_inputs_summary(state)),
        _readiness_line("MODEL / BACKEND", _model_backend_summary(state)),
        _readiness_line("KEYFRAME READINESS", _keyframe_readiness_summary(state)),
    ]


def _readiness_line(label: str, value: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(_shorten(value, 105), style=_value_style(value))
    return line


def _pipeline_route_summary(state: CockpitState) -> str:
    pipeline = state.header.pipeline or "unknown"
    mark = "✓" if pipeline != "unknown" else "○"
    return f"{mark} pipeline {pipeline} · current stage 09 Image / Keyframe Generation"


def _creative_inputs_summary(state: CockpitState) -> str:
    mode = state.header.mode or "unknown"
    style_hint = _style_hint(state)
    beat_plan = "present" if _data_dict(state, "selected_beat_plan") else "missing"
    scene_contracts = _scene_contract_count(state)
    mark = "✓" if mode != "unknown" and scene_contracts != "missing" else "○"
    return f"{mark} mode {mode} · style {style_hint} · hook/beat plan {beat_plan} · scene contracts {scene_contracts}"


def _prompt_inputs_summary(state: CockpitState) -> str:
    compiler = _prompt_compiler_status(state)
    audit = artifact_status(state, "prompt_audit.json")
    policy = _artifact_policy_status(state)
    mark = "✓" if compiler == "present" and audit == "present" else "○"
    return f"{mark} image prompt compiler {compiler} · prompt audit {audit} · artifact policy {policy}"


def _model_backend_summary(state: CockpitState) -> str:
    backend = _image_backend_status(state)
    session = state.session_mode or "unknown"
    mark = "✓" if backend not in {"unknown", "not_checked"} else "○"
    return f"{mark} image backend {backend} · run type {_run_type(state)} · session {session}"


def _keyframe_readiness_summary(state: CockpitState) -> str:
    counts = _image_job_counts(state)
    if counts["total"] == 0:
        return "○ total scenes 0 · finished 0 · generating 0 · queued/missing 0"
    mark = "▶" if counts["generating"] else "✓" if counts["queued_missing"] == 0 else "○"
    return (
        f"{mark} total scenes {counts['total']} · finished {counts['finished']} · "
        f"generating {counts['generating']} · queued/missing {counts['queued_missing']}"
    )


def _stage_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    if stage.view_type == "command_center":
        return _command_center_workspace(state, stage)
    if stage.view_type == "pipeline_select":
        return _pipeline_select_workspace(state, stage)
    if stage.view_type == "mode_style":
        return _mode_style_workspace(state, stage)
    if stage.view_type == "skills":
        return _skills_workspace(state, stage)
    if stage.stage_id == "04":
        return _strategy_workspace(state, stage)
    if stage.stage_id == "05":
        return _beat_workspace(state, stage)
    if stage.stage_id == "06":
        return _judge_workspace(state, stage)
    if stage.stage_id == "07":
        return _scene_contracts_workspace(state, stage)
    if stage.stage_id == "08":
        return _image_prompt_workspace(state, stage)
    if stage.stage_id == "10":
        return _keyframe_review_workspace(state, stage)
    if stage.stage_id == "11":
        return _ltx_prompt_workspace(state, stage)
    if stage.stage_id == "12":
        return _video_generation_workspace(state, stage)
    if stage.stage_id == "13":
        return _video_review_workspace(state, stage)
    if stage.stage_id == "14":
        return _assembly_workspace(state, stage)
    if stage.stage_id == "15":
        return _final_output_workspace(state, stage)
    return _placeholder_workspace(state, stage)


def _command_center_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_header(stage)
    workspace.append(
        _section_box(
            "CURRENT POSITION",
            _position_grid(
                (
                    ("Job ID", state.header.job_id),
                    ("Runs Root", str(state.data_source_path.parent)),
                    ("Session Mode", state.session_mode),
                    ("Run Type", _run_type(state)),
                    ("Data Source", state.header.artifact_mode),
                    ("Status", state.header.status),
                    ("Watch", "on" if state.watch_enabled else "off"),
                    ("Render Paused", state.workspace.render_paused),
                    ("Current Step", state.workspace.current_step),
                    ("Next", stage.next_action),
                )
            ),
        )
    )
    workspace.append(
        _section_box(
            "COMMAND COMPOSER",
            _position_grid(
                (
                    ("Topic", state.header.topic or "unknown"),
                    ("Format", f"{state.header.orientation} · {state.header.resolution}"),
                    ("Mode", state.header.mode or "unknown"),
                    ("Style", _style_hint(state)),
                    ("Duration", f"{state.header.duration}s" if state.header.duration else "unknown"),
                    ("Voice", _data_hint(state, "voice", "disabled")),
                    ("Music", _data_hint(state, "music", "disabled")),
                    ("Subtitles", _data_hint(state, "subtitles", "off")),
                    ("Storyboard", f"{state.header.scene_count or 'unknown'} scenes"),
                    ("Output", "keyframes + video plan"),
                )
            )
            + [
                _label_value_line("Action", "Run planned / disabled in V0.2"),
                _label_value_line("Safety", "read-only composer prototype"),
            ],
        )
    )
    workspace.append(
        _section_box(
            "COMMAND PREVIEW",
            [
                _plain_line("read-only preview; no submit action is wired"),
                _plain_line(_command_preview(state)),
            ],
        )
    )
    return workspace


def _command_preview(state: CockpitState) -> str:
    duration = f"{state.header.duration}s" if state.header.duration else "unknown"
    scene_count = str(state.header.scene_count or "unknown")
    topic = state.header.topic or "unknown"
    mode = state.header.mode or "unknown"
    return _shorten(
        (
            "creative-os compose"
            f" --topic \"{topic}\""
            f" --format {state.header.orientation}"
            f" --mode {mode}"
            f" --duration {duration}"
            f" --scenes {scene_count}"
        ),
        BOX_WIDTH - 2,
    )


def _pipeline_select_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_header(stage)
    rows = [
        _label_value_line("Current pipeline", state.header.pipeline),
        _label_value_line("Creative OS / storyboard pipeline", "available"),
        _label_value_line("Agent-Core run", "available" if state.run_type == "agent_core" else "not current"),
        _label_value_line("Generic content machine pipeline", "planned"),
        _label_value_line("Next", stage.next_action),
    ]
    workspace.append(_section_box("PIPELINE WÄHLEN", rows))
    workspace.append(_section_box("PIPELINE PURPOSE", [_plain_line("Storyboard and image/keyframe oriented Creative OS operator flow.")]))
    return workspace


def _mode_style_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_header(stage)
    workspace.append(
        _section_box(
            "MODE & STYLE",
            _position_grid(
                (
                    ("Mode", state.header.mode or "unknown"),
                    ("Style", _style_hint(state)),
                    ("Format", f"{state.header.orientation} · {state.header.resolution}"),
                    ("Topic", state.header.topic or "unknown"),
                    ("Duration", f"{state.header.duration}s" if state.header.duration else "unknown"),
                    ("Scenes", str(state.header.scene_count or "unknown")),
                    ("Style Lock", _data_hint(state, "visual_identity", "not_checked")),
                    ("Next", stage.next_action),
                )
            ),
        )
    )
    return workspace


def _skills_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_header(stage)
    workspace.append(
        _section_box(
            "SKILLS LADEN",
            [
                _label_value_line("Skill Health", f"{state.skill_health.mark} {state.skill_health.status}"),
                _label_value_line("Loaded", str(state.skill_health.loaded_count)),
                _label_value_line("Fallbacks", str(state.skill_health.fallback_count)),
                _label_value_line("Missing optional", str(state.skill_health.missing_optional_count)),
                _label_value_line("Blocking missing", str(state.skill_health.blocking_missing_count)),
            ],
        )
    )
    workspace.append(_section_box("SKILL GROUPS", _skill_group_lines(state)))
    return workspace


def _placeholder_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_header(stage)
    workspace.append(
        _section_box(
            stage.title.upper(),
            [
                _label_value_line("Status", _stage_artifact_summary(state, stage)),
                _label_value_line("Purpose", stage.short_description),
                _label_value_line("Next Action", stage.next_action),
                _plain_line("Detailed panel planned"),
            ],
        )
    )
    workspace.append(_section_box("EXPECTED ARTIFACTS", _artifact_lines(state, stage)))
    return workspace


def _strategy_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    strategy = _data_dict(state, "creative_strategy")
    director = _data_dict(state, "director_output")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "STRATEGY READOUT",
        [
            ("Hook", _first_value(strategy, "hook", "opening_hook", "attention_hook")),
            ("Core Idea", _first_value(strategy, "core_idea", "concept", "strategy")),
            ("Risks", _joined_value(strategy.get("risks") or strategy.get("risk_controls"))),
            ("Director", _first_value(director, "director_mode", "mode")),
        ],
        "creative_strategy.json / director_output.json",
        "Review strategy, hook and director constraints before beat planning.",
    )
    return workspace


def _beat_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    beat_plan = _data_dict(state, "selected_beat_plan")
    beats = beat_plan.get("beats") or beat_plan.get("selected_beats") or []
    workspace = _stage_detail_workspace(
        state,
        stage,
        "BEAT / HOOK PLAN",
        [
            ("Hook", _first_value(beat_plan, "hook", "opening_hook")),
            ("Beats", _count_or_unknown(beats)),
            ("Escalation", _first_value(beat_plan, "escalation", "middle")),
            ("Payoff", _first_value(beat_plan, "payoff", "ending")),
        ],
        "selected_beat_plan.json",
        "Inspect selected beat plan and prepare judge handoff.",
    )
    return workspace


def _judge_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    judge = _data_dict(state, "creative_judge") or _data_dict(state, "decision_log")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "CREATIVE JUDGE",
        [
            ("Decision", _first_value(judge, "decision", "status", "selected")),
            ("Rationale", _first_value(judge, "rationale", "reason", "notes")),
            ("Risks", _joined_value(judge.get("risks") or judge.get("issues"))),
            ("Selected", _first_value(judge, "selected_candidate", "candidate", "winner")),
        ],
        "creative_judge.json / decision_log.json",
        "Review selection rationale before scene contract expansion.",
    )
    return workspace


def _scene_contracts_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = Text()
    workspace.append("ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(_compact_section_box("CURRENT POSITION AND PIPELINE PATH", _scene_contract_position_lines(state), BOX_WIDTH))
    workspace.append(_scene_contract_columns(state))
    workspace.append(_compact_section_box("HANDOFF PATH", _scene_contract_handoff_lines(), BOX_WIDTH))
    return workspace


def _scene_contract_position_lines(state: CockpitState) -> list[Text]:
    return [
        _metric_row(
            (
                ("Current Step", "Scene Contracts"),
                ("Operator Focus", "lock scene-level production rules"),
                ("Render Paused", state.workspace.render_paused),
            ),
            value_width=30,
        ),
        _metric_row(
            (
                ("Last", "✓ 06 Creative Judge"),
                ("Next", "○ 08 Image Prompt Compiler"),
                ("Output", "scene_contracts.json"),
                ("Run", state.session_mode or _run_type(state)),
            ),
            value_width=28,
        ),
    ]


def _scene_contract_input_status_lines(state: CockpitState) -> list[Text]:
    return [
        _compact_status_line("✓", "Creative Strategy", _artifact_ready_demo_missing(state, "creative_strategy.json", "creative_strategy")),
        _compact_status_line("✓", "Selected Beat / Hook", _artifact_ready_demo_missing(state, "selected_beat_plan.json", "selected_beat_plan")),
        _compact_status_line("✓", "Creative Judge Decision", _creative_judge_status(state)),
        _compact_status_line("✓", "Mode / Style", _mode_style_contract_value(state)),
        _compact_status_line("✓", "Risk Policy", _risk_policy_status(state)),
        _compact_status_line("○", "Artifact Policy", _artifact_policy_contract_status(state)),
    ]


def _scene_contract_columns(state: CockpitState) -> Text:
    return _three_column_boxes(
        (
            ("A) CONTRACT INPUTS", _scene_contract_input_status_lines(state), SCENE_CONTRACT_A_WIDTH),
            ("B) SCENE CONTRACTS", _scene_contract_card_lines(state), SCENE_CONTRACT_B_WIDTH),
            ("C) OUTPUT PREVIEW / READINESS", _scene_contract_output_lines(state), SCENE_CONTRACT_C_WIDTH),
        )
    )


def _scene_contract_card_lines(state: CockpitState) -> list[Text]:
    scenes = _scene_contracts(state)
    if not scenes:
        return [_plain_line("No scene contracts available")]
    lines: list[Text] = []
    for index, scene in enumerate(scenes[:3], start=1):
        if index > 1:
            lines.append(_plain_line(""))
        lines.extend(_scene_contract_card(state, index, scene))
    return lines


def _scene_contract_card(state: CockpitState, index: int, scene: object) -> list[Text]:
    scene_id = _scene_value(scene, "scene_id", f"scene_{index:02d}")
    fixture = _fixture_scene_contract(scene_id)
    status = _scene_contract_status(state, index)
    marker = "✓" if "ready" in status else "▶" if "drafting" in status else "○"
    return [
        _scene_card_header_line(index, scene_id, marker, status),
        _compact_field_line("Visual Anchor", _scene_contract_value(scene, fixture, "visual_anchor", "anchor", "subject")),
        _compact_field_line("Environment", _scene_contract_value(scene, fixture, "environment", "setting", "location")),
        _compact_field_line("Visible Action", _scene_contract_value(scene, fixture, "action", "subject_action", "description")),
        _compact_field_line("Camera / Lighting", _scene_contract_camera_lighting(scene, fixture)),
        _compact_field_line("Allowed Visuals", _scene_contract_list_value(scene, fixture, "allowed_visuals", "allowed visuals")),
        _compact_field_line("Forbidden Visuals", _scene_contract_list_value(scene, fixture, "forbidden_visuals", "forbidden visuals")),
        _compact_field_line("Text/Glyph Risk", _scene_contract_text_risk(fixture)),
    ]


def _scene_contract_output_lines(state: CockpitState) -> list[Text]:
    scenes = _scene_contracts(state)
    artifact_status_value = _artifact_ready_demo_missing(state, "scene_contracts.json", "scene_contracts")
    completeness = "✓ complete" if len(scenes) >= 3 else f"○ {len(scenes)}/3 contracts"
    return [
        _plain_line("Output Preview (JSON)"),
        _plain_line("{"),
        _plain_line('  "file": "scene_contracts.json",'),
        _plain_line(f'  "scene_count": {len(scenes) or state.header.scene_count or 0},'),
        _plain_line('  "continuity_rule": "jungle sunrise progression",'),
        _plain_line('  "text_policy": "no readable text",'),
        _plain_line('  "ready_for": "image_prompt_compiler"'),
        _plain_line("}"),
        _plain_line(""),
        _compact_status_line("✓", "scene_contracts.json", artifact_status_value),
        _compact_status_line("✓", "scenes", f"{len(scenes) or state.header.scene_count or 0} contracts"),
        _compact_status_line("✓", "contract completeness", completeness),
        _compact_status_line("○", "artifact policy attached", _artifact_policy_contract_status(state)),
        _compact_status_line("○", "ready for Stage 08", "✓ Image Prompt Compiler" if scenes else "○ missing contracts"),
    ]


def _scene_contract_handoff_lines() -> list[Text]:
    return [
        _plain_line("06 Creative Judge -> Scene Contracts ACTIVE -> 08 Image Prompt Compiler"),
    ]


def _image_prompt_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = Text()
    workspace.append("ACTIVE WORKSPACE - PROMPT COMPILER\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(stage.short_description + "\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    workspace.append(_section_box("CURRENT POSITION", _compiler_position_lines(state)))
    workspace.append(_section_box("COMPILER SCOPE / OVERVIEW", _compiler_scope_lines(state)))
    workspace.append(_active_section_box("IMAGE COMPILER (ACTIVE)", _image_compiler_lines(state)))
    workspace.append(_section_box("COMPILER FAMILY / BRANCHES", _compiler_family_lines()))
    workspace.append(_section_box("OUTPUT / NEXT", _compiler_output_lines(state)))
    return workspace


def _compiler_position_lines(state: CockpitState) -> list[Text]:
    return _position_grid(
        (
            ("Current Step", "PromptCompiler"),
            ("Operator Focus", "compile final prompts"),
            ("Render Paused", state.workspace.render_paused),
            ("Last Passed", "✓ 07 Scene Contracts"),
            ("Stage", "08 Prompt Compiler"),
            ("Active Branch", "Image Compiler"),
            ("Output Ready", _compiler_output_readiness(state)),
            ("Next", "09 Image / Keyframe Generation"),
        )
    )


def _compiler_scope_lines(state: CockpitState) -> list[Text]:
    scenes = _compiler_scenes(state)
    assets = sum(1 for _line, ok in state.artifacts.lines if ok)
    return [
        _scope_metric_line(
            (
                ("Scenes", str(len(scenes) or state.header.scene_count or 0)),
                ("Assets", str(assets or "not_checked")),
                ("Style Signals", str(_style_signal_count(state))),
                ("Skill Sources", str(state.skill_health.loaded_count or "not_checked")),
                ("Output Targets", "image · video · audio · music"),
            )
        ),
        _readiness_line("Input Readiness", _compiler_input_readiness(state)),
        _readiness_line("Output Readiness", _compiler_output_readiness(state)),
    ]


def _image_compiler_lines(state: CockpitState) -> list[Text]:
    return [
        _readiness_line("Status", "✓ Image Compiler active"),
        _readiness_line("Pipeline / Route", _compiler_route_summary(state)),
        _plain_line(""),
        _compiler_subtitle("SCENE CONTRACT INPUTS"),
        *_scene_contract_input_lines(state),
        _plain_line(""),
        _compiler_subtitle("SCENE PROMPT SUMMARIES"),
        *_scene_prompt_summary_lines(state),
        _plain_line(""),
        _compiler_subtitle("FINAL PROMPT PAYLOAD (JSON PREVIEW)"),
        *_final_prompt_payload_lines(state),
        _plain_line(""),
        _compiler_subtitle("MODEL RULES / ARTIFACT POLICY"),
        *_model_rule_lines(state),
    ]


def _compiler_route_summary(state: CockpitState) -> str:
    pipeline = state.header.pipeline or "unknown"
    mark = "✓" if pipeline != "unknown" else "○"
    return f"{mark} pipeline {pipeline} · current stage 08 Prompt Compiler / Image Compiler"


def _compiler_family_lines() -> list[Text]:
    return [
        _label_value_line("Image Compiler", "✓ active"),
        _label_value_line("Video Compiler", "○ queued / later · output video_prompts.json"),
        _label_value_line("Audio Compiler", "○ queued / optional · output audio_prompts.json"),
        _label_value_line("Music Compiler", "○ queued / optional · output music_prompts.json"),
    ]


def _image_prompt_card_lines(state: CockpitState) -> list[Text]:
    scenes = _compiler_scenes(state)
    if not scenes:
        return [_plain_line("No scene contracts available for image prompt compilation")]
    lines: list[Text] = []
    for index, scene in enumerate(scenes[:3], start=1):
        if index > 1:
            lines.append(_plain_line(""))
        scene_id = _scene_value(scene, "scene_id", f"scene_{index:02d}")
        lines.extend(_image_prompt_card(state, scene, scene_id))
    return lines


def _image_prompt_card(state: CockpitState, scene: object, scene_id: str) -> list[Text]:
    prompt = _compiled_prompt_preview(state, scene_id)
    return [
        _label_value_line("Scene ID", scene_id),
        _label_value_line("Contract Source", _contract_source_status(state, scene_id)),
        _label_value_line("Visual Anchor", _scene_field(scene, "visual_anchor", "anchor", "subject", default=_visual_anchor_hint(state))),
        _label_value_line("Environment", _scene_field(scene, "environment", "setting", "location")),
        _label_value_line("Action", _scene_field(scene, "action", "subject_action", "description")),
        _label_value_line("Camera / Lighting", _camera_lighting_value(scene)),
        _label_value_line("Positive Prompt", prompt),
        _label_value_line("Artifact Bans", _artifact_bans_value(state)),
        _label_value_line("Output Target", f"{scene_id}_image_prompt.json"),
        _label_value_line("Status", _image_prompt_status(state, scene_id, prompt)),
    ]


def _scope_metric_line(metrics: tuple[tuple[str, str], ...]) -> Text:
    line = Text()
    for index, (label, value) in enumerate(metrics):
        if index:
            line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        line.append(_shorten(value, 18), style=_value_style(value))
    return line


def _compiler_subtitle(value: str) -> Text:
    line = Text()
    line.append(value, style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    return line


def _scene_contract_input_lines(state: CockpitState) -> list[Text]:
    first = _compiler_scenes(state)[0] if _compiler_scenes(state) else {}
    return [
        _label_value_line("Style", _style_hint(state)),
        _label_value_line("Mode", state.header.mode or "unknown"),
        _label_value_line("Tone", _scene_field(first, "tone", "mood", default="not_checked")),
        _label_value_line("Lighting", _scene_field(first, "lighting", "light", default="not_checked")),
        _label_value_line("Camera", _scene_field(first, "camera", "camera_plan", "camera_motion", default="not_checked")),
        _label_value_line("Composition", _scene_field(first, "composition", "framing", default="not_checked")),
        _label_value_line("Resolution", state.header.resolution or "unknown"),
        _label_value_line("Aspect", state.header.orientation or "unknown"),
    ]


def _scene_prompt_summary_lines(state: CockpitState) -> list[Text]:
    scenes = _compiler_scenes(state)
    if not scenes:
        return [_plain_line("No prompt summaries available")]
    lines: list[Text] = []
    for index, scene in enumerate(scenes[:3], start=1):
        scene_id = _scene_value(scene, "scene_id", f"scene_{index:02d}")
        prompt = _compiled_prompt_preview(state, scene_id)
        status = _scene_prompt_status_for_summary(state, index, prompt)
        confidence = _scene_prompt_confidence(index, status)
        line = Text()
        line.append(f"{scene_id:<9} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        line.append(f"{status:<10} ", style=_job_status_style("generating" if status == "validating" else "ready"))
        line.append(_shorten(prompt if prompt != "not_checked" else _prompt_fallback_summary(state, scene_id), 70), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
        line.append(f"  conf: {confidence}", style=style(TEXT_SUCCESS if status == "compiled" else TEXT_ACTIVE, bg=BG_WORKSPACE, bold=True))
        lines.append(line)
    return lines


def _final_prompt_payload_lines(state: CockpitState) -> list[Text]:
    scenes = [_scene_value(scene, "scene_id", f"scene_{index:02d}") for index, scene in enumerate(_compiler_scenes(state)[:3], start=1)]
    scene_list = ", ".join(scenes) or "none"
    return [
        _plain_line("{"),
        _plain_line(f'  "project": "{state.header.job_id}", "mode": "{state.header.mode or "unknown"}",'),
        _plain_line(f'  "format": "{state.header.orientation} / {state.header.resolution}", "scenes": [{scene_list}],'),
        _plain_line('  "output": "prompt_payload_compiled.json", "handoff": "09 Image / Keyframe Generation"'),
        _plain_line("}"),
    ]


def _model_rule_lines(state: CockpitState) -> list[Text]:
    return [
        _label_value_line("zimage rules", _model_rules_status(state)),
        _label_value_line("avoid text artifacts", _artifact_policy_active_status(state)),
        _label_value_line("no readable text", _policy_rule_status(state, "no_readable_text")),
        _label_value_line("no screens/paper/labels", _policy_rule_status(state, "no_screens_paper_labels")),
        _label_value_line("clean visual subject", _policy_rule_status(state, "clean_visual_subject")),
        _label_value_line("Stage 09 handoff", "✓ ready for Stage 09" if _prompt_compiler_status(state) == "present" else "○ missing prompt output"),
    ]


def _compiler_output_lines(state: CockpitState) -> list[Text]:
    image_prompts = "ready" if _prompt_compiler_status(state) == "present" else "missing"
    audit = "ready" if artifact_status(state, "prompt_audit.json") == "present" else "missing"
    return [
        _label_value_line("prompt_payload_compiled.json", "in progress" if image_prompts == "ready" else "missing"),
        _label_value_line("image_prompts.json", image_prompts),
        _label_value_line("prompt_audit.json", audit),
        _label_value_line("next stage", "09 Image / Keyframe Generation"),
    ]


def _keyframe_review_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    reviews = _data_list(state, "keyframe_review")
    decision = _data_dict(state, "stage6_review_decision")
    statuses = [str(item.get("status") or item.get("review_status") or "unknown") for item in reviews if isinstance(item, dict)]
    workspace = _stage_detail_workspace(
        state,
        stage,
        "KEYFRAME REVIEW",
        [
            ("Reviewed", str(len(reviews)) if reviews else "unknown"),
            ("Passed", str(sum(1 for status in statuses if status == "passed")) if statuses else "unknown"),
            ("Needs Review", str(sum(1 for status in statuses if status == "needs_review")) if statuses else "unknown"),
            ("Rejected", str(sum(1 for status in statuses if status == "rejected")) if statuses else "unknown"),
            ("Reviewer", _first_value(decision, "reviewer", "provider", "review_provider")),
        ],
        "keyframe_review.json / stage6_review_decision.json",
        "Confirm keyframes before LTX motion prompt compilation.",
    )
    return workspace


def _ltx_prompt_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    prompts = _data_list(state, "ltx_motion_prompts")
    audit = _data_dict(state, "ltx_prompt_audit")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "LTX MOTION PROMPT COMPILER",
        [
            ("Motion Prompts", str(len(prompts)) if prompts else "unknown"),
            ("Audit", _first_value(audit, "overall", "status", "result")),
            ("Render Started", _first_value(audit, "render_started", "video_started")),
            ("Artifacts", "ltx_motion_prompts.json / ltx_prompt_audit.json"),
        ],
        "ltx_motion_prompts.json / ltx_prompt_audit.json",
        "Inspect motion prompt audit before any video-generation gate.",
    )
    return workspace


def _video_generation_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    manifest = _data_dict(state, "ltx_video_takes_manifest")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "LTX VIDEO GENERATION",
        [
            ("Takes Manifest", artifact_status(state, "ltx_video_takes_manifest.json")),
            ("Videos", _first_value(manifest, "videos", "takes", "outputs")),
            ("Render", "not_started"),
            ("Gate", "read-only cockpit; no render action"),
        ],
        "ltx_video_takes_manifest.json / video files",
        "Wait for an explicit render gate outside this cockpit view.",
    )
    return workspace


def _video_review_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    review = _data_dict(state, "video_review")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "VIDEO REVIEW",
        [
            ("Review Artifact", artifact_status(state, "video_review.json")),
            ("Status", _first_value(review, "status", "overall", "result")),
            ("Findings", _joined_value(review.get("findings") or review.get("issues"))),
            ("Reviewer", _first_value(review, "reviewer", "provider")),
        ],
        "video_review.json",
        "Detailed video review panel planned after video takes exist.",
    )
    return workspace


def _assembly_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    workspace = _stage_detail_workspace(
        state,
        stage,
        "FINAL ASSEMBLY",
        [
            ("Final MP4", _final_mp4_status(state)),
            ("Voice", artifact_status(state, "voice.wav")),
            ("Music", artifact_status(state, "music.wav")),
            ("Subtitles", artifact_status(state, "subtitles.srt")),
            ("Overlay", artifact_status(state, "overlay.json")),
        ],
        "final.mp4 / voice / music / subtitles / overlay",
        "Assemble only via explicit pipeline tooling, not from this cockpit.",
    )
    return workspace


def _final_output_workspace(state: CockpitState, stage: StageDefinition) -> Text:
    verdict = _data_dict(state, "final_quality_verdict")
    workspace = _stage_detail_workspace(
        state,
        stage,
        "FINAL OUTPUT",
        [
            ("Final Verdict", _first_value(verdict, "verdict", "status", "overall")),
            ("Final MP4", _final_mp4_status(state)),
            ("Postable", _first_value(verdict, "postable", "is_postable")),
            ("Output Status", _first_value(verdict, "output_status", "result")),
        ],
        "final_quality_verdict.json / final.mp4",
        "Review final output readiness; no publishing action is available here.",
    )
    return workspace


def _stage_detail_workspace(
    state: CockpitState,
    stage: StageDefinition,
    section_title: str,
    rows: list[tuple[str, str]],
    expected_output: str,
    next_action: str,
) -> Text:
    workspace = _stage_header(stage)
    workspace.append(
        _section_box(
            section_title,
            [
                _label_value_line("Current Status", _stage_artifact_summary(state, stage)),
                _label_value_line("Purpose", stage.short_description),
                *[_label_value_line(label, value or "unknown") for label, value in rows],
                _label_value_line("Expected Output", expected_output),
                _label_value_line("Next Action", next_action),
            ],
        )
    )
    workspace.append(_section_box("ARTIFACTS", _artifact_lines(state, stage)))
    return workspace


def _stage_header(stage: StageDefinition) -> Text:
    workspace = Text()
    workspace.append(f"{stage.stage_id} {stage.title.upper()}\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(f"{stage.short_description}\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return workspace


def _prompt_job_lines(state: CockpitState) -> list[Text]:
    data = state.workspace
    if not data.scenes:
        line = Text()
        line.append("○ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        line.append("No prompt/image jobs available", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return [line]
    lines: list[Text] = []
    scene_count = min(3, len(data.scenes))
    selected = int(getattr(state, "selected_image_job", 2) or 2)
    selected = max(1, min(scene_count, selected))
    expanded_jobs = set(getattr(state, "expanded_image_jobs", (2,)) or ())
    if not any(1 <= index <= scene_count for index in expanded_jobs):
        expanded_jobs.add(selected)
    for index, scene in enumerate(data.scenes[:3], start=1):
        if index > 1:
            lines.append(_job_gap_line())
        lines.extend(_render_prompt_job(state, index, scene, selected=index == selected, expanded=index in expanded_jobs))
    return lines


def _render_prompt_job(state: CockpitState, index: int, scene: object, *, selected: bool, expanded: bool) -> list[Text]:
    scene_id = _scene_value(scene, "scene_id", f"scene_{index:02d}")
    source = _scene_value(scene, "keyframe", "missing")
    summary = _scene_value(scene, "summary", "unknown")
    title = _scene_value(scene, "title", scene_id)
    job_status = _image_job_status(state, index, scene)
    caret = "v" if expanded else ">"

    lines = [
        _job_card_border(top=True),
        _job_card_top(index, scene_id, source, job_status, caret, selected),
        _job_card_summary(title, scene_id, summary, job_status),
        _job_card_source(source),
    ]
    if expanded:
        lines.extend(_job_detail_lines(state, index, scene, _scene_prompt(state, scene_id) or summary, source, job_status))
    lines.append(_job_card_border(top=False))
    return lines


def _job_card_border(*, top: bool) -> Text:
    line = Text()
    left = "╭" if top else "╰"
    right = "╮" if top else "╯"
    line.append(left + "─" * JOB_CARD_INNER_WIDTH + right, style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return line


def _job_card_top(index: int, scene_id: str, source: str, job_status: str, caret: str, selected: bool) -> Text:
    content = Text()
    marker = "▸" if selected else " "
    content.append(f"{marker} ", style=style(TEXT_LABEL if selected else TEXT_MUTED, bg=BG_WORKSPACE, bold=True))
    content.append(f"{_shorten(_preview_slot(source, job_status), JOB_CARD_PREVIEW_WIDTH):<{JOB_CARD_PREVIEW_WIDTH}}", style=_preview_style(job_status))
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    title = f"Image {index} / {scene_id}"
    content.append(f"{_shorten(title, JOB_CARD_MAIN_WIDTH):<{JOB_CARD_MAIN_WIDTH}}", style=style(TEXT_MAIN, bg=BG_WORKSPACE, bold=True))
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append(f"{_shorten(job_status, JOB_CARD_STATUS_WIDTH - 4):>{JOB_CARD_STATUS_WIDTH - 4}} ", style=_job_status_style(job_status))
    content.append(f"[{caret}]", style=style(TEXT_LABEL if selected else TEXT_MUTED, bg=BG_WORKSPACE, bold=True))
    return _job_card_line(content)


def _job_card_summary(title: str, scene_id: str, summary: str, job_status: str) -> Text:
    content = Text()
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append(" " * JOB_CARD_PREVIEW_WIDTH, style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    main_text = f"{title} · {summary}" if title != scene_id else summary
    content.append(_shorten(main_text, JOB_CARD_MAIN_WIDTH), style=style(TEXT_MAIN if job_status != "in queue" else TEXT_MUTED, bg=BG_WORKSPACE))
    return _job_card_line(content)


def _job_card_source(source: str) -> Text:
    content = Text()
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append(" " * JOB_CARD_PREVIEW_WIDTH, style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append("keyframe: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    content.append(_shorten(_basename(source), JOB_CARD_MAIN_WIDTH - 10), style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return _job_card_line(content)


def _job_detail_lines(state: CockpitState, index: int, scene: object, summary: str, source: str, status: str) -> list[Text]:
    progress = _image_job_progress(state, index, scene, status)
    title = f"Generating Image {index}" if status == "generating" else f"Image {index}"
    lines = [
        _job_card_line(Text("")),
        _job_detail_title(title),
        _job_detail_prompt(_scene_optional(scene, "prompt") or summary),
        _job_detail_status(progress, status),
        _job_detail_meta(source, progress),
    ]
    return lines


def _job_detail_title(title: str) -> Text:
    content = Text()
    content.append("  " + " " * JOB_CARD_PREVIEW_WIDTH + "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append(title, style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    return _job_card_line(content)


def _job_detail_prompt(prompt: str) -> Text:
    content = Text()
    content.append("  " + " " * JOB_CARD_PREVIEW_WIDTH + "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append("Prompt: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    content.append(_shorten(prompt, 86), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return _job_card_line(content)


def _job_detail_status(progress: dict[str, object], status: str) -> Text:
    content = Text()
    content.append("  " + " " * JOB_CARD_PREVIEW_WIDTH + "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    content.append("Status: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    percent = progress["percent"]
    if percent is None:
        content.append(status, style=_job_status_style(status))
    else:
        content.append(_progress_bar(str(percent)), style=style(TEXT_ACTIVE if int(float(percent)) < 100 else TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True))
        content.append(f" {int(float(percent))}%", style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return _job_card_line(content)


def _job_detail_meta(source: str, progress: dict[str, object]) -> Text:
    content = Text()
    content.append("  " + " " * JOB_CARD_PREVIEW_WIDTH + "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    elapsed = str(progress["elapsed"] or "not_checked")
    backend = str(progress["backend"] or "not_checked")
    content.append(f"elapsed {elapsed} · backend {backend}", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    source_name = _basename(source)
    if source_name:
        content.append(f" · source {source_name}", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return _job_card_line(content)


def _progress_line(progress: str, *, demo: bool = False) -> Text:
    line = Text()
    label = "demo progress " if demo else "progress "
    line.append(label, style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    try:
        percent = max(0, min(100, int(float(progress))))
    except ValueError:
        line.append(_shorten(progress, 80), style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return line
    line.append(f"{_progress_bar(str(percent))} {percent}%", style=style(TEXT_ACTIVE if percent < 100 else TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True))
    return line


def _progress_bar(progress: str) -> str:
    try:
        percent = max(0, min(100, int(float(progress))))
    except ValueError:
        percent = 0
    filled = round(percent / 5)
    return "[" + "█" * filled + "░" * (20 - filled) + "]"


def _job_card_line(content: Text) -> Text:
    line = Text()
    border_style = style(TEXT_MUTED, bg=BG_WORKSPACE)
    line.append("│ ", style=border_style)
    clipped = _truncate_text(content, JOB_CARD_INNER_WIDTH - 2)
    line.append_text(clipped)
    line.append(" " * max(0, JOB_CARD_INNER_WIDTH - len(clipped.plain) - 1), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    line.append("│", style=border_style)
    return line


def _job_gap_line() -> Text:
    line = Text()
    line.append("", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return line


def _preview_label(source: str) -> str:
    basename = _basename(source)
    if basename in {"missing", "unknown", "not_checked"}:
        return "empty preview"
    return basename


def _preview_slot(source: str, status: str) -> str:
    basename = _basename(source)
    if status in {"finished", "ready"} and basename not in {"missing", "unknown", "not_checked"}:
        return f"[img] {basename}"
    if status == "generating":
        return "[work] preview"
    return "[empty] slot"


def _preview_style(status: str) -> str:
    if status in {"finished", "ready"}:
        return style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    if status == "generating":
        return style(TEXT_ACTIVE, bg=BG_WORKSPACE, bold=True)
    return style(TEXT_MUTED, bg=BG_WORKSPACE)


def _image_job_status(state: CockpitState, index: int, scene: object) -> str:
    explicit = _scene_optional(scene, "job_status") or _scene_optional(scene, "queue_state")
    if explicit in {"finished", "ready", "generating", "in queue"}:
        return explicit
    if state.session_mode == "fixture/demo" and state.run_type == "creative_os":
        return ("ready", "generating", "in queue")[min(index, 3) - 1]
    state_label = _scene_value(scene, "state_label", "ready")
    output_status = _scene_value(scene, "status", "unknown")
    status = _status_for_job(state_label, output_status)
    if status in {"finished", "ready"}:
        return status
    source = _scene_value(scene, "keyframe", "")
    if source and source not in {"missing", "unknown"}:
        return "ready"
    return "in queue"


def _image_job_progress(state: CockpitState, index: int, scene: object, status: str) -> dict[str, object]:
    progress = _scene_optional(scene, "progress_percent")
    elapsed = _scene_optional(scene, "elapsed")
    backend = _scene_optional(scene, "backend") or _scene_optional(scene, "generator")
    if progress:
        return {"percent": progress, "elapsed": elapsed, "backend": backend, "demo": False}
    if state.session_mode == "fixture/demo" and status == "generating":
        return {"percent": 62, "elapsed": "00:18 demo", "backend": "zimage_http demo", "demo": True}
    if status in {"finished", "ready"}:
        return {"percent": 100, "elapsed": elapsed or "done", "backend": backend, "demo": False}
    return {"percent": None, "elapsed": elapsed or "waiting", "backend": backend, "demo": False}


def _scene_contract_count(state: CockpitState) -> str:
    scenes = _scene_contracts(state)
    if scenes:
        return f"{len(scenes)} present"
    if state.header.scene_count:
        return f"{state.header.scene_count} expected"
    return "missing"


def _scene_contracts(state: CockpitState) -> list[object]:
    return _data_list(state, "scene_contracts") or _data_list(state, "keyframe_contracts")


def _compiler_scenes(state: CockpitState) -> list[object]:
    return _scene_contracts(state) or _data_list(state, "zimage_prompts") or _model_prompt_scenes(state)


def _present_missing(items: list[object], source: str) -> str:
    return f"✓ present ({source})" if items else "○ missing"


def _artifact_ready_demo_missing(state: CockpitState, artifact: str, data_key: str) -> str:
    if artifact_status(state, artifact) == "present" or state.inspection.data.get(data_key):
        return "demo" if state.session_mode == "fixture/demo" else "✓ ready"
    return "○ missing"


def _mode_style_contract_value(state: CockpitState) -> str:
    mode = state.header.mode or "unknown"
    style_hint = _style_hint(state)
    if style_hint == "unknown":
        return f"{mode} / not_checked"
    return f"✓ {mode} / {style_hint}"


def _risk_policy_status(state: CockpitState) -> str:
    policy = _artifact_policy_status(state)
    if policy != "not_checked":
        return f"✓ {policy}"
    if state.session_mode == "fixture/demo":
        return "demo text_safe / artifact_avoidance"
    return "not_checked"


def _artifact_policy_contract_status(state: CockpitState) -> str:
    policy = _artifact_policy_status(state)
    if policy != "not_checked":
        return f"✓ {policy}"
    if state.session_mode == "fixture/demo":
        return "demo artifact_avoidance"
    return "○ missing"


def _fixture_scene_contract(scene_id: str) -> dict[str, str]:
    fixtures = {
        "scene_01": {
            "visual_anchor": "misty canopy reveal",
            "environment": "sunrise jungle canopy",
            "action": "slow opening reveal through leaves",
            "camera": "controlled push-in",
            "lighting": "warm shafts through mist",
            "allowed_visuals": "leaves, vines, depth haze",
            "forbidden_visuals": "text, logos, extra animals",
            "text_risk": "low / no readable text",
        },
        "scene_02": {
            "visual_anchor": "suspense jungle trail",
            "environment": "narrow dense path",
            "action": "cautious forward motion",
            "camera": "low forward glide",
            "lighting": "filtered green-gold light",
            "allowed_visuals": "wet leaves, foliage depth, trail",
            "forbidden_visuals": "captions, UI text, crowd elements",
            "text_risk": "medium / avoid captions",
        },
        "scene_03": {
            "visual_anchor": "golden path payoff",
            "environment": "opening jungle corridor",
            "action": "reveal into brighter destination",
            "camera": "steady forward push",
            "lighting": "strong golden end light",
            "allowed_visuals": "path, sun rays, layered foliage",
            "forbidden_visuals": "signage, duplicate subjects, overlays",
            "text_risk": "medium / avoid signage",
        },
    }
    return fixtures.get(scene_id, {})


def _scene_contract_status(state: CockpitState, index: int) -> str:
    if state.session_mode == "fixture/demo":
        return ("✓ contract ready", "▶ contract drafting", "○ queued")[min(index, 3) - 1]
    return "✓ ready" if index <= len(_scene_contracts(state)) else "○ missing"


def _scene_contract_value(scene: object, fixture: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = _scene_optional(scene, key)
        if value:
            return value
    for key in keys:
        normalized = key.replace(" ", "_")
        if fixture.get(normalized):
            return fixture[normalized]
    return "not_checked"


def _scene_contract_camera_lighting(scene: object, fixture: dict[str, str]) -> str:
    camera = _scene_contract_value(scene, fixture, "camera", "camera_plan", "camera_motion")
    lighting = _scene_contract_value(scene, fixture, "lighting", "light", "mood")
    if camera == "not_checked" and lighting == "not_checked":
        return "not_checked"
    return f"{camera} / {lighting}"


def _scene_contract_list_value(scene: object, fixture: dict[str, str], key: str, label: str) -> str:
    value = _scene_optional(scene, key) or _scene_optional(scene, label)
    if value:
        return value
    return fixture.get(key) or "not_checked"


def _scene_contract_text_risk(fixture: dict[str, str]) -> str:
    return fixture.get("text_risk") or "not_checked"


def _compiler_input_readiness(state: CockpitState) -> str:
    ready = sum(
        1
        for value in (
            _present_missing(_scene_contracts(state), "scene_contracts.json"),
            _creative_judge_status(state),
            _model_rules_status(state),
        )
        if value.startswith("✓")
    )
    return f"✓ {ready}/3 ready" if ready == 3 else f"○ {ready}/3 ready"


def _compiler_output_readiness(state: CockpitState) -> str:
    if _prompt_compiler_status(state) == "present":
        return "✓ image prompts ready"
    return "○ image prompts missing"


def _style_signal_count(state: CockpitState) -> int:
    values = {
        state.header.mode,
        state.header.topic,
        state.header.orientation,
        state.header.resolution,
        _style_hint(state),
        _artifact_policy_status(state),
    }
    return sum(1 for value in values if value and value not in {"unknown", "not_checked"})


def _creative_judge_status(state: CockpitState) -> str:
    judge = _data_dict(state, "creative_judge") or _data_dict(state, "decision_log") or _data_dict(state, "stage6_review_decision")
    status = _first_value(judge, "overall_status", "decision", "status", "result")
    if status in {"passed", "approved", "selected"}:
        return f"✓ approved ({status})"
    if status != "unknown":
        return f"○ {status}"
    return "○ missing"


def _style_lock_status(state: CockpitState) -> str:
    style_hint = _style_hint(state)
    if style_hint != "unknown":
        return f"✓ present ({style_hint})"
    value = _data_hint(state, "visual_identity", "not_checked")
    if value != "not_checked":
        return f"✓ present ({value})"
    return "○ missing"


def _artifact_policy_active_status(state: CockpitState) -> str:
    policy = _artifact_policy_status(state)
    if policy != "not_checked":
        return f"✓ active ({policy})"
    return "○ missing"


def _model_rules_status(state: CockpitState) -> str:
    backend = _image_backend_status(state)
    if backend not in {"unknown", "not_checked"}:
        return f"✓ loaded ({backend})"
    if _prompt_compiler_status(state) == "present":
        return "✓ loaded (prompt artifact)"
    return "○ missing"


def _prompt_compiler_status(state: CockpitState) -> str:
    if artifact_status(state, "zimage_prompts.json") == "present" or artifact_status(state, "model_prompts.json") == "present":
        return "present"
    prompts = _data_list(state, "zimage_prompts") or _model_prompt_scenes(state)
    return "present" if prompts else "missing"


def _artifact_policy_status(state: CockpitState) -> str:
    audit = _data_dict(state, "prompt_audit")
    for key in ("artifact_policy", "avoid_text_artifacts", "text_artifacts", "policy"):
        value = audit.get(key)
        if value not in (None, "", [], {}):
            return _format_value(value)
    return "not_checked"


def _image_backend_status(state: CockpitState) -> str:
    for scene in state.workspace.scenes:
        backend = _scene_optional(scene, "backend") or _scene_optional(scene, "generator")
        if backend:
            return backend
    provider = _prompt_provider(state)
    if provider == "zimage":
        return "zimage_http" if state.session_mode == "fixture/demo" else "zimage"
    if provider != "unknown":
        return provider
    for _label, value in state.system_status.rows:
        if "Image Backend" in _label and value not in {"? not_checked", "not_checked", "unknown"}:
            return value.removeprefix("✓ ").removeprefix("? ").strip()
    return "not_checked"


def _contract_source_status(state: CockpitState, scene_id: str) -> str:
    for scene in _scene_contracts(state):
        if _scene_value(scene, "scene_id", "") == scene_id:
            return "scene_contracts.json"
    return "missing"


def _scene_field(scene: object, *keys: str, default: str = "not_checked") -> str:
    for key in keys:
        value = _scene_optional(scene, key)
        if value:
            return value
    return default


def _visual_anchor_hint(state: CockpitState) -> str:
    topic = state.header.topic or "unknown"
    mode = state.header.mode or "unknown"
    if topic == "unknown" and mode == "unknown":
        return "not_checked"
    return f"{topic} / {mode}"


def _camera_lighting_value(scene: object) -> str:
    camera = _scene_field(scene, "camera", "camera_plan", "camera_motion")
    lighting = _scene_field(scene, "lighting", "light", "mood")
    if camera == "not_checked" and lighting == "not_checked":
        return "not_checked"
    return f"{camera} / {lighting}"


def _compiled_prompt_preview(state: CockpitState, scene_id: str) -> str:
    for prompt in _data_list(state, "zimage_prompts"):
        if isinstance(prompt, dict) and str(prompt.get("scene_id") or "") == scene_id:
            for key in ("positive_prompt", "model_prompt", "prompt", "prompt_text"):
                value = prompt.get(key)
                if value:
                    return str(value)
            return "not_checked"
    prompt = _scene_prompt(state, scene_id)
    return prompt or "not_checked"


def _artifact_bans_value(state: CockpitState) -> str:
    policy = _artifact_policy_status(state)
    if policy != "not_checked":
        return policy
    return "avoid text artifacts; no readable text; no screens/paper/labels"


def _image_prompt_status(state: CockpitState, scene_id: str, prompt: str) -> str:
    if state.session_mode == "fixture/demo" and _prompt_compiler_status(state) == "present":
        return "demo"
    if prompt not in {"missing", "not_checked", "unknown"}:
        return "ready"
    for prompt_item in _data_list(state, "zimage_prompts") + _model_prompt_scenes(state):
        if isinstance(prompt_item, dict) and str(prompt_item.get("scene_id") or "") == scene_id:
            return "ready"
    return "missing"


def _scene_prompt_status_for_summary(state: CockpitState, index: int, prompt: str) -> str:
    if state.session_mode == "fixture/demo":
        return "validating" if index == 3 else "compiled"
    if prompt not in {"missing", "not_checked", "unknown"}:
        return "compiled"
    return "missing"


def _scene_prompt_confidence(index: int, status: str) -> str:
    if status == "missing":
        return "n/a"
    return ("0.94", "0.93", "0.91")[min(index, 3) - 1]


def _prompt_fallback_summary(state: CockpitState, scene_id: str) -> str:
    topic = state.header.topic or "visual subject"
    style_hint = _style_hint(state)
    return f"{topic}, {style_hint}, clean subject, no readable text, scene {scene_id} demo summary"


def _policy_rule_status(state: CockpitState, key: str) -> str:
    policy = _artifact_policy_status(state)
    if policy != "not_checked":
        return f"✓ {policy}"
    return {
        "no_readable_text": "not_checked",
        "no_screens_paper_labels": "not_checked",
        "clean_visual_subject": "not_checked",
    }.get(key, "not_checked")


def _image_job_counts(state: CockpitState) -> dict[str, int]:
    scenes = tuple(state.workspace.scenes[:3])
    counts = {"total": len(scenes), "finished": 0, "generating": 0, "queued_missing": 0}
    for index, scene in enumerate(scenes, start=1):
        status = _image_job_status(state, index, scene)
        if status in {"finished", "ready", "read-only"}:
            counts["finished"] += 1
        elif status == "generating":
            counts["generating"] += 1
        else:
            counts["queued_missing"] += 1
    return counts


def _final_mp4_status(state: CockpitState) -> str:
    value = getattr(state.workspace, "final_mp4", None)
    if value:
        return str(value)
    for label, status in state.system_status.rows:
        if label == "Final MP4":
            return status
    for line, _ok in state.artifacts.lines:
        if "final.mp4" in line:
            return line.replace("final.mp4", "").strip() or "unknown"
    return "not_checked"


def _director_mode(state: CockpitState) -> str:
    value = getattr(state.workspace, "director_mode", None)
    if value:
        return str(value)
    if state.header.mode:
        return state.header.mode
    return "unknown"


def _run_type(state: CockpitState) -> str:
    value = getattr(state.workspace, "run_type", None)
    if value:
        return str(value)
    if state.run_type:
        return state.run_type
    return "unknown"


def _scene_prompt(state: CockpitState, scene_id: str) -> str | None:
    prompt_data = state.inspection.data.get("model_prompts")
    if not isinstance(prompt_data, dict):
        prompt_path = state.data_source_path / "model_prompts.json"
        if not prompt_path.exists():
            return None
        try:
            loaded = json.loads(prompt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        prompt_data = loaded if isinstance(loaded, dict) else {}
    scenes = prompt_data.get("scenes")
    if not isinstance(scenes, list):
        return None
    for scene in scenes:
        if not isinstance(scene, dict) or str(scene.get("scene_id") or "") != scene_id:
            continue
        for key in ("model_prompt", "prompt", "prompt_text"):
            value = scene.get(key)
            if value:
                return str(value)
    return None


def _label_value_line(label: str, value: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(_shorten(value, 100), style=_value_style(value))
    return line


def _plain_line(value: str) -> Text:
    line = Text()
    line.append(_shorten(value, 120), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return line


def _artifact_lines(state: CockpitState, stage: StageDefinition) -> list[Text]:
    if not stage.artifacts:
        return [_plain_line("No required artifact for this V0.1 view")]
    return [_label_value_line(artifact, artifact_status(state, artifact)) for artifact in stage.artifacts]


def _stage_artifact_summary(state: CockpitState, stage: StageDefinition) -> str:
    if not stage.artifacts:
        return "read-only"
    present = sum(1 for artifact in stage.artifacts if artifact_status(state, artifact) == "present")
    if present == len(stage.artifacts):
        return "passed"
    if present:
        return f"{present}/{len(stage.artifacts)} artifacts present"
    return "pending"


def _style_hint(state: CockpitState) -> str:
    data = state.inspection.data
    for key in ("creative_strategy", "normalized_job", "intent_route"):
        value = data.get(key)
        if isinstance(value, dict):
            style_value = value.get("style") or value.get("visual_style") or value.get("style_intent")
            if style_value:
                return str(style_value)
    return "unknown"


def _data_hint(state: CockpitState, key: str, default: str) -> str:
    for value in state.inspection.data.values():
        if isinstance(value, dict) and value.get(key):
            return str(value[key])
    return default


def _data_dict(state: CockpitState, key: str) -> dict[str, object]:
    value = state.inspection.data.get(key)
    return value if isinstance(value, dict) else {}


def _data_list(state: CockpitState, key: str) -> list[object]:
    value = state.inspection.data.get(key)
    return value if isinstance(value, list) else []


def _model_prompt_scenes(state: CockpitState) -> list[object]:
    value = state.inspection.data.get("model_prompts")
    if isinstance(value, dict) and isinstance(value.get("scenes"), list):
        return value["scenes"]
    return []


def _first_value(data: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return _format_value(value)
    return "unknown"


def _joined_value(value: object) -> str:
    if value in (None, "", [], {}):
        return "unknown"
    return _format_value(value)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value[:3]) or "unknown"
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:3]:
            parts.append(f"{key}={_format_value(item)}")
        return ", ".join(parts) or "unknown"
    return str(value)


def _count_or_unknown(value: object) -> str:
    if isinstance(value, (list, tuple, dict)):
        return str(len(value))
    if value in (None, ""):
        return "unknown"
    return "1"


def _prompt_provider(state: CockpitState) -> str:
    prompts = _data_list(state, "zimage_prompts")
    for prompt in prompts:
        if isinstance(prompt, dict):
            value = prompt.get("provider") or prompt.get("backend") or prompt.get("generator")
            if value:
                return str(value)
    if state.run_type == "creative_os" and artifact_status(state, "zimage_prompts.json") == "present":
        return "zimage"
    if state.run_type == "agent_core" and artifact_status(state, "model_prompts.json") == "present":
        return "agent_core"
    return "unknown"


def _skill_group_lines(state: CockpitState) -> list[Text]:
    skill_match = state.inspection.data.get("skill_match")
    if not isinstance(skill_match, dict):
        return [
            _label_value_line("core", "unknown"),
            _label_value_line("model", "unknown"),
            _label_value_line("style", "unknown"),
            _label_value_line("fallback", "unknown"),
            _label_value_line("missing", "unknown"),
        ]
    return [
        _label_value_line("core", _skill_group_value(skill_match, ("core", "loaded_core", "core_skills"))),
        _label_value_line("model", _skill_group_value(skill_match, ("model", "models", "model_skills"))),
        _label_value_line("style", _skill_group_value(skill_match, ("style", "styles", "style_skills"))),
        _label_value_line("fallback", _skill_group_value(skill_match, ("fallback", "fallbacks", "fallback_skills"))),
        _label_value_line("missing", _skill_group_value(skill_match, ("missing", "missing_optional", "blocking_missing"))),
        _label_value_line("reasons", _skill_group_value(skill_match, ("reasons", "reason", "rationale"))),
    ]


def _skill_group_value(skill_match: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = skill_match.get(key)
        if value:
            if isinstance(value, list):
                return ", ".join(str(item) for item in value) or "unknown"
            if isinstance(value, dict):
                return ", ".join(str(item) for item in value) or "unknown"
            return str(value)
    return "unknown"


def _missing_run_text(state: CockpitState) -> Text:
    workspace = rows_text(
        [
            ("Current Step", "Run not found"),
            ("Searched", str(state.data_source_path)),
            ("Hint", "use --runs-root for fixture/demo data or create a real run first"),
            ("Watch", "on" if state.watch_enabled else "off"),
            ("Render paused", "unknown"),
            ("Run type", "missing"),
            ("Final MP4", "unknown"),
            ("Director Mode", "unknown"),
        ],
        label_width=17,
        bg=BG_WORKSPACE,
    )
    workspace.append("\nRun not found\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(f"searched: {state.data_source_path}\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    workspace.append(
        "hint: use --runs-root for fixture/demo data or create a real run first\n",
        style=style(TEXT_MUTED, bg=BG_WORKSPACE),
    )
    return workspace


def _append_stage_outputs(workspace: Text, outputs: tuple[str, ...]) -> None:
    if not outputs:
        workspace.append("○ not_checked\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return
    for index, output in enumerate(outputs):
        marker_style = style(TEXT_SUCCESS if output.startswith("✓") else TEXT_MUTED, bg=BG_WORKSPACE)
        workspace.append(output, style=marker_style)
        workspace.append("\n" if index == len(outputs) - 1 else "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))


def _status_grid(rows: list[tuple[str, str]]) -> Text:
    text = Text()
    left_width = 17
    value_width = 42
    for index in range(0, len(rows), 2):
        left_label, left_value = rows[index]
        right_label, right_value = rows[index + 1] if index + 1 < len(rows) else ("", "")
        text.append(f"{left_label.upper():<{left_width}} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        text.append(_shorten(left_value, value_width).ljust(value_width), style=_value_style(left_value))
        if right_label:
            text.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
            text.append(f"{right_label.upper():<{left_width}} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
            text.append(_shorten(right_value, value_width), style=_value_style(right_value))
        text.append("\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return text


def _value_style(value: str) -> str:
    if value.startswith("✓"):
        return style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    if value.startswith("▶") or "generating" in value:
        return style(TEXT_ACTIVE, bg=BG_WORKSPACE, bold=True)
    if value.startswith("?") or value.startswith("-") or value in {"unknown", "not_checked"}:
        return style(TEXT_MUTED, bg=BG_WORKSPACE)
    return style(TEXT_MUTED if value.startswith("○") else TEXT_MAIN, bg=BG_WORKSPACE)


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _metric_row(metrics: tuple[tuple[str, str], ...], *, value_width: int) -> Text:
    line = Text()
    for index, (label, value) in enumerate(metrics):
        if index:
            line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        line.append(_shorten(value, value_width), style=_value_style(value))
    return line


def _compact_status_line(marker: str, label: str, value: str) -> Text:
    marker_style = TEXT_SUCCESS if marker == "✓" else TEXT_ACTIVE if marker == "▶" else TEXT_MUTED
    line = Text()
    line.append(f"{marker} ", style=style(marker_style, bg=BG_WORKSPACE, bold=True))
    line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(_shorten(value, 42), style=_value_style(value))
    return line


def _compact_field_line(label: str, value: str) -> Text:
    line = Text()
    line.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(_shorten(value, 45), style=_value_style(value))
    return line


def _scene_card_header_line(index: int, scene_id: str, marker: str, status: str) -> Text:
    marker_style = TEXT_SUCCESS if marker == "✓" else TEXT_ACTIVE if marker == "▶" else TEXT_MUTED
    line = Text()
    line.append(f"{index} ", style=style(marker_style, bg=BG_WORKSPACE, bold=True))
    line.append(f"{scene_id:<11}", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(status, style=_value_style(status))
    return line


def _compact_section_box(title: str, lines: list[Text], width: int) -> Text:
    box = Text()
    border_style = style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True)
    inner_width = width - 3
    title_text = _shorten(title, width - 6)
    box.append("╭─ ", style=border_style)
    box.append(title_text, style=border_style)
    box.append(" " + "─" * max(0, width - len(title_text) - 5) + "╮\n", style=border_style)
    for line in lines:
        content = _truncate_text(line, inner_width)
        box.append("│ ", style=border_style)
        box.append_text(content)
        box.append(" " * max(0, inner_width - len(content.plain)), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
        box.append("│\n", style=border_style)
    box.append("╰" + "─" * (width - 2) + "╯\n", style=border_style)
    return box


def _three_column_boxes(columns: tuple[tuple[str, list[Text], int], ...]) -> Text:
    rendered = [_box_plain_lines(title, lines, width) for title, lines, width in columns]
    height = max(len(lines) for lines in rendered)
    for lines, (_title, _content, width) in zip(rendered, columns):
        while len(lines) < height:
            lines.insert(-1, "│ " + " " * (width - 3) + "│")

    output = Text()
    for row_index in range(height):
        for column_index, lines in enumerate(rendered):
            output.append(lines[row_index], style=style(TEXT_LABEL if row_index in {0, height - 1} else TEXT_MAIN, bg=BG_WORKSPACE))
            if column_index < len(rendered) - 1:
                output.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        output.append("\n", style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return output


def _box_plain_lines(title: str, lines: list[Text], width: int) -> list[str]:
    inner_width = width - 3
    title_text = _shorten(title, width - 6)
    result = [
        "╭─ " + title_text + " " + "─" * max(0, width - len(title_text) - 5) + "╮",
    ]
    for line in lines:
        plain = _shorten(line.plain, inner_width)
        result.append("│ " + plain.ljust(inner_width) + "│")
    result.append("╰" + "─" * (width - 2) + "╯")
    return result


def _section_box(title: str, lines: list[Text]) -> Text:
    box = Text()
    border_style = style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True)
    box.append("╭─ ", style=border_style)
    box.append(_shorten(title, BOX_WIDTH - 6), style=border_style)
    box.append(" " + "─" * max(0, BOX_WIDTH - len(title) - 5) + "╮\n", style=border_style)
    for line in lines:
        _append_box_line(box, line)
    box.append("╰" + "─" * BOX_WIDTH + "╯\n", style=border_style)
    return box


def _active_section_box(title: str, lines: list[Text]) -> Text:
    box = Text()
    border_style = style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    box.append("╭─ ", style=border_style)
    box.append(_shorten(title, BOX_WIDTH - 6), style=border_style)
    box.append(" " + "─" * max(0, BOX_WIDTH - len(title) - 5) + "╮\n", style=border_style)
    for line in lines:
        _append_active_box_line(box, line)
    box.append("╰" + "─" * BOX_WIDTH + "╯\n", style=border_style)
    return box


def _append_active_box_line(box: Text, line: Text) -> None:
    content = _truncate_text(line, BOX_WIDTH - 2)
    box.append("│ ", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True))
    box.append_text(content)
    visible_len = len(content.plain)
    box.append(" " * max(0, BOX_WIDTH - visible_len - 1), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    box.append("│\n", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True))


def _append_box_line(box: Text, line: Text) -> None:
    content = _truncate_text(line, BOX_WIDTH - 2)
    box.append("│ ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    box.append_text(content)
    visible_len = len(content.plain)
    box.append(" " * max(0, BOX_WIDTH - visible_len - 1), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    box.append("│\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))


def _position_grid(fields: tuple[tuple[str, str], ...]) -> list[Text]:
    rows: list[Text] = []
    for start in range(0, len(fields), 4):
        group = fields[start : start + 4]
        rows.append(_cell_row(group, labels=True))
        rows.append(_cell_row(group, labels=False))
        if start + 4 < len(fields):
            rows.append(_blank_position_row())
    return rows


def _cell_row(fields: tuple[tuple[str, str], ...], *, labels: bool) -> Text:
    line = Text()
    cell_width = 29
    for index, (label, value) in enumerate(fields):
        if index:
            line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        text = label.upper() if labels else value
        cell_style = style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True) if labels else _value_style(value)
        line.append(f"{_shorten(text, cell_width):<{cell_width}}", style=cell_style)
    return line


def _blank_position_row() -> Text:
    line = Text()
    return line


def _status_for_job(state_label: str, output_status: str) -> str:
    lower = f"{state_label} {output_status}".lower()
    if "missing" in lower:
        return "missing"
    if "ready" in lower:
        return "ready"
    if "present" in lower or "passed" in lower or "finished" in lower:
        return "finished"
    if "waiting" in lower or "queued" in lower:
        return "queued"
    if "plan" in lower:
        return "read-only"
    return "unknown"


def _job_status_style(status: str) -> str:
    if status in {"finished", "ready", "read-only"}:
        return style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    if status == "generating":
        return style(TEXT_ACTIVE, bg=BG_WORKSPACE, bold=True)
    if status == "missing":
        return style(TEXT_MUTED, bg=BG_WORKSPACE)
    return style(TEXT_MUTED, bg=BG_WORKSPACE)


def _truncate_text(text: Text, limit: int) -> Text:
    if len(text.plain) <= limit:
        return text
    truncated = Text()
    remaining = max(0, limit - 1)
    for span_text, span_style in _text_spans(text):
        if remaining <= 0:
            break
        chunk = span_text[:remaining]
        truncated.append(chunk, style=span_style)
        remaining -= len(chunk)
    truncated.append("…", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return truncated


def _pad_text(text: Text, width: int) -> Text:
    padded = Text()
    padded.append_text(text)
    padded.append(" " * max(0, width - len(text.plain)), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return padded


def _text_spans(text: Text) -> list[tuple[str, str | None]]:
    spans: list[tuple[str, str | None]] = []
    plain = text.plain
    if not text.spans:
        return [(plain, None)]
    position = 0
    for span in sorted(text.spans, key=lambda item: item.start):
        if span.start > position:
            spans.append((plain[position : span.start], None))
        spans.append((plain[span.start : span.end], span.style))
        position = span.end
    if position < len(plain):
        spans.append((plain[position:], None))
    return spans


def _scene_value(scene: object, key: str, default: str) -> str:
    if isinstance(scene, dict):
        return str(scene.get(key) or default)
    return str(getattr(scene, key, default) or default)


def _scene_optional(scene: object, key: str) -> str | None:
    if isinstance(scene, dict):
        value = scene.get(key)
    else:
        value = getattr(scene, key, None)
    if value in (None, ""):
        return None
    return str(value)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else "missing"
