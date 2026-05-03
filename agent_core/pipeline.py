from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_core.schemas import CheckpointRecord, PipelineDefinition, PipelineStepDefinition
from agent_core.utils import utc_now_iso


PIPELINE_DEFS_DIR = Path(__file__).resolve().parent / "pipeline_defs"
DEFAULT_PIPELINE_ID = "simple_video_v1"
STOP_AFTER_POINTS = ("scene_plan", "model_prompts", "storyboard")


class ApprovalGateBlocked(RuntimeError):
    def __init__(self, checkpoint: CheckpointRecord, approval_path: Path) -> None:
        self.checkpoint = checkpoint
        self.approval_path = approval_path
        super().__init__(
            f"checkpoint {checkpoint.checkpoint_id} requires approval file {approval_path}"
        )


def load_pipeline_definition(
    pipeline_id: str = DEFAULT_PIPELINE_ID,
    *,
    definitions_dir: str | Path | None = None,
) -> PipelineDefinition:
    root = Path(definitions_dir) if definitions_dir is not None else PIPELINE_DEFS_DIR
    path = root / f"{pipeline_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"pipeline definition not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineDefinition.model_validate(payload)


def pipeline_step(definition: PipelineDefinition, step_id: str) -> PipelineStepDefinition | None:
    for step in definition.steps:
        if step.step_id == step_id:
            return step
    return None


def checkpoint_for_step(definition: PipelineDefinition, step_id: str) -> CheckpointRecord:
    step = pipeline_step(definition, step_id)
    if step is None:
        return CheckpointRecord(
            checkpoint_id=step_id,
            stage=step_id,
            status="pending",
            blocking=False,
            reason="ad-hoc checkpoint outside pipeline definition",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    now = utc_now_iso()
    return CheckpointRecord(
        checkpoint_id=step.checkpoint_id or step.step_id,
        stage=step.stage,
        status="pending",
        blocking=step.blocking,
        approval_required=step.approval_required,
        created_at=now,
        updated_at=now,
        metadata={
            "pipeline_step_id": step.step_id,
            "required_inputs": list(step.required_inputs),
            "produced_artifacts": list(step.produced_artifacts),
            "required_skills": list(step.required_skills),
            "optional": step.optional,
        },
    )


def approval_file_path(job_dir: Path, definition: PipelineDefinition, checkpoint_id: str) -> Path:
    approval_dir = definition.approval_policy.approval_dir or "approvals"
    return job_dir / approval_dir / f"{checkpoint_id}.json"


def read_approval_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    approved = payload.get("approved")
    if str(approved).strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    return payload


def approval_gates_enabled(job_metadata: dict[str, Any] | None) -> bool:
    value = (job_metadata or {}).get("approval_gates_enabled")
    return str(value).strip().lower() in {"1", "true", "yes", "on", "manual", "manual_file"}


def pipeline_dry_run_enabled(job_metadata: dict[str, Any] | None) -> bool:
    value = (job_metadata or {}).get("pipeline_dry_run")
    return str(value).strip().lower() in {"1", "true", "yes", "on", "dry_run"}


def normalize_stop_after(job_metadata: dict[str, Any] | None) -> str | None:
    value = str((job_metadata or {}).get("stop_after") or "").strip().lower().replace("-", "_")
    aliases = {
        "scene": "scene_plan",
        "sceneplan": "scene_plan",
        "scene_plan": "scene_plan",
        "plan": "scene_plan",
        "prompts": "model_prompts",
        "model_prompt": "model_prompts",
        "model_prompts": "model_prompts",
        "prompt_audit": "model_prompts",
        "storyboard": "storyboard",
        "storyboard_plan": "storyboard",
    }
    normalized = aliases.get(value)
    if not normalized:
        return None
    return normalized


def stop_after_reached(job_metadata: dict[str, Any] | None, point: str) -> bool:
    return normalize_stop_after(job_metadata) == point
