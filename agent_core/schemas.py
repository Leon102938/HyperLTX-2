from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JobPhase = Literal[
    "received",
    "validated",
    "planned",
    "voice_generated",
    "video_generated",
    "assembled",
    "done",
    "failed",
]

StepStatus = Literal["pending", "planned", "running", "succeeded", "skipped", "failed"]
BackendKind = Literal["voice", "video", "music", "storyboard"]
VideoMode = Literal["auto", "text_only", "storyboard_reference", "keyframe_conditioned"]
RenderMode = Literal["text_only", "storyboard_reference", "keyframe_conditioned"]
DirectorMode = Literal["llm_augmented", "rule_based_fallback"]
TakeReviewStatus = Literal["passed", "failed", "rejected", "selected"]
TakeValidationStatus = Literal["passed", "failed", "rejected"]
KeyframeReviewStatus = Literal["passed", "needs_review", "failed", "rejected", "selected"]
ImageValidationStatus = Literal["passed", "failed", "rejected"]
CheckpointStatus = Literal["pending", "passed", "failed", "needs_review", "skipped"]
ApprovalMode = Literal["auto", "manual_file", "disabled"]


class PipelineStepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    stage: str
    required_inputs: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    checkpoint_id: str | None = None
    approval_required: bool = False
    blocking: bool = True
    optional: bool = False
    notes: list[str] = Field(default_factory=list)


class PipelineRetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = 1
    retryable_statuses: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PipelineApprovalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ApprovalMode = "auto"
    require_files_when_enabled: bool = True
    approval_dir: str = "approvals"
    notes: list[str] = Field(default_factory=list)


class PipelineDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_id: str
    mode: str = "default"
    steps: list[PipelineStepDefinition]
    required_inputs: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    stage_roles: dict[str, str] = Field(default_factory=dict)
    checkpoints: list[str] = Field(default_factory=list)
    default_policy: dict[str, Any] = Field(default_factory=dict)
    retry_policy: PipelineRetryPolicy = Field(default_factory=PipelineRetryPolicy)
    approval_policy: PipelineApprovalPolicy = Field(default_factory=PipelineApprovalPolicy)
    notes: list[str] = Field(default_factory=list)


class CreativeStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    mode_id: str
    style_id: str | None = None
    user_intent: str
    creative_goal: str
    platform: str | None = None
    target_platform: str | None = None
    hook_pattern: str | None = None
    pacing: dict[str, Any] = Field(default_factory=dict)
    creative_freedom: str | None = None
    continuity_mode: str | None = None
    audience_intent: str | None = None
    audience_feel: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    anti_goals: list[str] = Field(default_factory=list)
    motif_families: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeatPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_plan_id: str
    beats: list[dict[str, Any]] = Field(default_factory=list)
    scene_roles: dict[str, str] = Field(default_factory=dict)
    timing_intent: str | None = None
    escalation_logic: str | None = None
    payoff: str | None = None
    selected_motif_families: list[str] = Field(default_factory=list)
    selected_shot_recipes: list[str] = Field(default_factory=list)
    transition_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualDirection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction_id: str
    visual_identity: str
    motif_family: str | None = None
    shot_recipe: str | None = None
    lighting: str | None = None
    camera_language: str | None = None
    motion_language: str | None = None
    movement: str | None = None
    composition_rules: list[str] = Field(default_factory=list)
    object_count_policy: str | None = None
    human_action_policy: str | None = None
    avoid_risks: list[str] = Field(default_factory=list)
    allowed_visuals: list[str] = Field(default_factory=list)
    forbidden_visuals: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPromptPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_plan_id: str
    backend_prompt_policy: dict[str, str] = Field(default_factory=dict)
    positive_model_prompt: str | None = None
    negative_model_prompt: str | None = None
    zimage_prompt_sent: str | None = None
    ltx_positive_prompt_sent: str | None = None
    ltx_negative_prompt_sent: str | None = None
    warnings: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    loaded_model_skills: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_plan_id: str
    provider: str | None = None
    technical_checks: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    creative_quality_checks: list[str] = Field(default_factory=list)
    platform_fit_checks: list[str] = Field(default_factory=list)
    artifact_checks: list[str] = Field(default_factory=list)
    rejection_rules: list[str] = Field(default_factory=list)
    selection_policy: str | None = None
    skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    stage: str
    decision: str
    reason: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    pipeline_id: str | None = None
    version: str = "g2_decision_log_v1"
    decisions: list[DecisionLogEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    stage: str
    status: CheckpointStatus = "pending"
    blocking: bool = False
    reason: str | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    related_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    approval_required: bool = False
    approved_by: str | None = None
    approved_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str
    path: str
    origin: str
    exists: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: StepStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    backend_name: str | None = None
    backend_job_id: str | None = None
    output_path: str | None = None
    output_url: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class BackendCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: BackendKind
    available: bool
    phase1_enabled: bool = True
    transport: str = "internal"
    supported_pipelines: list[str] = Field(default_factory=list)
    supported_orientations: list[str] = Field(default_factory=list)
    supported_resolution_labels: list[str] = Field(default_factory=list)
    supports_image_conditioning: bool = False
    notes: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_name: str
    success: bool
    status: StepStatus
    backend_name: str
    backend_job_id: str | None = None
    output_path: str | None = None
    output_url: str | None = None
    duration_sec: float | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    job_id: str | None = None
    idea: str = ""
    script: str = ""
    duration_sec: float | None = None
    format: str | None = None
    orientation: str | None = None
    resolution: str = "standard"
    use_voice: bool = True
    voice_id: str = "Ryan"
    use_music: bool = False
    use_storyboard: bool = False
    video_mode: VideoMode = "auto"
    style: str = "cinematic"
    extra_llm_instruction: str = ""
    pipeline_preference: str = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("idea", "script", "style", "extra_llm_instruction")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0 or value > 600:
            raise ValueError("duration_sec must be between 0 and 600 seconds")
        return round(value, 2)

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: str) -> str:
        normalized = value.lower().strip()
        valid_labels = {"draft", "standard", "high"}
        if normalized in valid_labels:
            return normalized

        parts = normalized.split("x")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            width, height = int(parts[0]), int(parts[1])
            if width < 256 or height < 256:
                raise ValueError("custom resolution must be at least 256x256")
            if width % 64 != 0 or height % 64 != 0:
                raise ValueError("custom resolution must use width and height divisible by 64 in Phase 1")
            return f"{width}x{height}"

        raise ValueError("resolution must be one of draft, standard, high or an explicit WxH value")

    @field_validator("orientation", "format")
    @classmethod
    def validate_orientation(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        normalized = value.lower().strip()
        if normalized not in {"landscape", "portrait", "square"}:
            raise ValueError("orientation/format must be landscape, portrait or square")
        return normalized

    @field_validator("pipeline_preference")
    @classmethod
    def validate_pipeline_preference(cls, value: str) -> str:
        normalized = value.lower().strip()
        allowed = {"auto", "ti2vid", "a2vid", "fast", "balanced", "quality"}
        if normalized not in allowed:
            raise ValueError(f"pipeline_preference must be one of {sorted(allowed)}")
        return normalized

    @field_validator("video_mode")
    @classmethod
    def validate_video_mode(cls, value: str) -> str:
        normalized = value.lower().strip()
        allowed = {"auto", "text_only", "storyboard_reference", "keyframe_conditioned"}
        if normalized not in allowed:
            raise ValueError(f"video_mode must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def finalize(self) -> "JobInput":
        if not self.idea and not self.script:
            raise ValueError("At least one of idea or script must be provided")

        if self.orientation and self.format and self.orientation != self.format:
            raise ValueError("format and orientation must match when both are provided")

        resolved = self.orientation or self.format or "landscape"
        self.orientation = resolved
        self.format = resolved

        if not self.use_voice:
            self.voice_id = ""

        return self

    @property
    def primary_text(self) -> str:
        return self.script or self.idea


class ProductionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: BackendKind | str
    adapter_name: str | None = None
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class ShotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: str
    scene_id: str
    index: int
    description: str
    target_duration_sec: float
    num_frames: int
    prompt_text: str
    narration_text: str | None = None
    narration_start_sec: float | None = None
    narration_end_sec: float | None = None
    render_params: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class VariationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variation_id: str
    scene_id: str
    variation_index: int
    shot_type: str
    camera_style: str | None = None
    camera_motion: str | None = None
    framing_hint: str
    prompt_delta: str | None = None
    prompt_variant_text: str
    style_bias: str | None = None
    creative_intent: str | None = None
    prompt_build_metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_camera_hint(self) -> "VariationPlan":
        if not self.camera_style and not self.camera_motion:
            raise ValueError("variation requires at least camera_style or camera_motion")
        return self


class StoryboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    enabled: bool = False
    required: bool = False
    candidate_count: int = 0
    preferred_variation_id: str | None = None
    preferred_variation_index: int | None = None
    priority_rule: str | None = None
    selection_mode: str = "preferred_variation_then_first_valid"
    notes: list[str] = Field(default_factory=list)


class KeyframeCandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    scene_id: str
    candidate_index: int
    variation_id: str | None = None
    variation_index: int | None = None
    shot_type: str | None = None
    prompt_text: str
    width: int
    height: int
    priority_rank: int = 0
    relation_type: str = "scene_variation"
    render_params: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ImageValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_status: ImageValidationStatus
    passed: bool = False
    file_exists: bool = False
    file_size_bytes: int | None = None
    minimum_size_bytes: int | None = None
    image_open_ok: bool = False
    width: int | None = None
    height: int | None = None
    format_name: str | None = None
    color_mode: str | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SelectedKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    scene_id: str
    candidate_index: int
    variation_id: str | None = None
    variation_index: int | None = None
    shot_type: str | None = None
    output_path: str | None = None
    output_url: str | None = None
    selected_by_rule: str | None = None
    selection_reason: str | None = None
    technical_status: str | None = None
    validation: ImageValidationReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TakePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_id: str
    scene_id: str
    take_index: int
    variation_id: str | None = None
    variation_index: int | None = None
    shot_type: str | None = None
    camera_style: str | None = None
    camera_motion: str | None = None
    framing_hint: str | None = None
    prompt_variant_text: str | None = None
    style_bias: str | None = None
    creative_intent: str | None = None
    prompt_build_metadata: dict[str, Any] = Field(default_factory=dict)
    seed: int
    prompt_text: str
    video_mode: VideoMode = "auto"
    render_mode: RenderMode = "text_only"
    fallback_strategy: str = "text_only"
    render_params: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TakeValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_status: TakeValidationStatus
    passed: bool = False
    file_exists: bool = False
    file_size_bytes: int | None = None
    minimum_size_bytes: int | None = None
    ffprobe_ok: bool = False
    decode_ok: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    duration_delta_sec: float | None = None
    codec_name: str | None = None
    format_name: str | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    expected_fps: float | None = None
    expected_duration_sec: float | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TakeRetryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    source_take_id: str
    retry_take_id: str
    retry_index: int
    seed: int
    reason: str
    source_variation_id: str | None = None
    retry_variation_id: str | None = None


class TakeResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_id: str
    scene_id: str
    take_index: int
    variation_id: str | None = None
    variation_index: int | None = None
    shot_type: str | None = None
    camera_style: str | None = None
    camera_motion: str | None = None
    framing_hint: str | None = None
    prompt_variant_text: str | None = None
    style_bias: str | None = None
    creative_intent: str | None = None
    prompt_build_metadata: dict[str, Any] = Field(default_factory=dict)
    seed: int
    video_mode: VideoMode = "auto"
    render_mode: RenderMode = "text_only"
    fallback_strategy: str = "text_only"
    fallback_reason: str | None = None
    status: StepStatus
    review_status: TakeReviewStatus = "failed"
    output_path: str | None = None
    output_url: str | None = None
    duration_sec: float | None = None
    selected: bool = False
    attempt_number: int = 1
    is_retry: bool = False
    retry_of_take_id: str | None = None
    retry_reason: str | None = None
    validation: TakeValidationReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class KeyframeCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    scene_id: str
    candidate_index: int
    variation_id: str | None = None
    variation_index: int | None = None
    shot_type: str | None = None
    status: StepStatus
    review_status: KeyframeReviewStatus = "failed"
    output_path: str | None = None
    output_url: str | None = None
    selected: bool = False
    validation: ImageValidationReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class VariationDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    shot_type: str
    intent: str
    camera_style: str | None = None
    camera_motion: str | None = None
    framing_hint: str
    prompt_delta: str
    style_bias: str | None = None


class CreativeBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept: str
    hook: str
    audience_intent: str
    narrative_arc: str
    emotional_arc: str
    payoff: str
    notes: list[str] = Field(default_factory=list)


class StyleLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_label: str
    visual_identity: str
    color_palette: str
    lighting: str
    camera_language: str
    texture: str
    pacing: str
    keep: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class PromptGuidance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opening_shot: str
    visual_language: list[str] = Field(default_factory=list)
    camera_cues: list[str] = Field(default_factory=list)
    prompt_rules: list[str] = Field(default_factory=list)
    negative_cues: list[str] = Field(default_factory=list)


class SceneIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    scene_index: int
    narrative_role: str
    hook_focus: str
    emotional_beat: str
    visual_goal: str
    shot_intent: str
    opening_emphasis: bool = False
    transition_note: str | None = None
    prompt_keywords: list[str] = Field(default_factory=list)
    variation_directives: list[VariationDirective] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DirectorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: DirectorMode
    active: bool = True
    fallback_reason: str | None = None
    llm_active: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_endpoint: str | None = None
    creative_brief: CreativeBrief
    style_lock: StyleLock
    prompt_guidance: PromptGuidance
    scene_intents: list[SceneIntent] = Field(default_factory=list)
    character_notes: list[str] = Field(default_factory=list)
    voice_notes: list[str] = Field(default_factory=list)
    world_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    index: int
    title: str
    description: str
    target_duration_sec: float
    num_frames: int
    prompt_text: str
    narration_text: str | None = None
    narration_start_sec: float | None = None
    narration_end_sec: float | None = None
    scene_intent: SceneIntent | None = None
    prompt_build_metadata: dict[str, Any] = Field(default_factory=dict)
    video_mode: VideoMode = "auto"
    render_mode: RenderMode = "text_only"
    fallback_strategy: str = "text_only"
    render_params: dict[str, Any] = Field(default_factory=dict)
    shots: list[ShotPlan] = Field(default_factory=list)
    variations: list[VariationPlan] = Field(default_factory=list)
    storyboard_config: StoryboardConfig | None = None
    keyframe_candidates: list[KeyframeCandidatePlan] = Field(default_factory=list)
    selected_keyframe: SelectedKeyframe | None = None
    takes: list[TakePlan] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProductionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    orientation: str
    resolution_label: str
    width: int
    height: int
    render_profile: str
    selected_pipeline: str
    requested_duration_sec: float | None = None
    target_duration_sec: float
    estimated_voice_duration_sec: float | None = None
    actual_voice_duration_sec: float | None = None
    prompt_text: str
    director_output: DirectorOutput | None = None
    warnings: list[str] = Field(default_factory=list)
    rules_applied: list[str] = Field(default_factory=list)
    scenes: list[ScenePlan] = Field(default_factory=list)
    steps: list[ProductionStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobPhase
    current_phase: JobPhase
    created_at: str
    updated_at: str
    plan_version: int = 0
    steps: dict[str, StepRunRecord] = Field(default_factory=dict)
    pipeline_id: str | None = None
    checkpoints: dict[str, CheckpointRecord] = Field(default_factory=dict)
    current_checkpoint_id: str | None = None
    blocked_by_checkpoint_id: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    result_path: str | None = None


class ResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    success: bool
    final_phase: JobPhase
    message: str
    planned_duration_sec: float | None = None
    actual_voice_duration_sec: float | None = None
    actual_video_duration_sec: float | None = None
    actual_final_duration_sec: float | None = None
    output_final_path: str | None = None
    output_video_path: str | None = None
    output_audio_path: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    backend_runs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
