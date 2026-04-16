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
TakeReviewStatus = Literal["passed", "failed", "rejected", "selected"]
TakeValidationStatus = Literal["passed", "failed", "rejected"]
KeyframeReviewStatus = Literal["passed", "failed", "rejected", "selected"]
ImageValidationStatus = Literal["passed", "failed", "rejected"]


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
