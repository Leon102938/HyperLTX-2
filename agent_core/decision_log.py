from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.schemas import DecisionLog, DecisionLogEntry, JobInput, ProductionPlan
from agent_core.utils import utc_now_iso, write_json


def decision_entry(
    decision_id: str,
    stage: str,
    decision: str,
    *,
    reason: str | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionLogEntry:
    return DecisionLogEntry(
        decision_id=decision_id,
        stage=stage,
        decision=decision,
        reason=reason,
        inputs=inputs or {},
        outputs=outputs or {},
        created_at=utc_now_iso(),
        metadata=metadata or {},
    )


def build_initial_decision_log(
    job: JobInput,
    plan: ProductionPlan,
    *,
    pipeline_id: str | None = None,
    skill_trace: dict[str, Any] | None = None,
    checkpoint_trace: dict[str, Any] | None = None,
) -> DecisionLog:
    metadata = dict(plan.metadata or {})
    skill_trace = skill_trace or {}
    mode_id = str(metadata.get("mode_id") or "")
    style_id = str(metadata.get("style_id") or "")
    backend_prompt_policy = metadata.get("backend_prompt_policy") or {}
    stop_after = str((job.metadata or {}).get("stop_after") or "")

    motif_families = metadata.get("motif_families") or []
    selected_shot_recipes = sorted(
        {
            str(scene.prompt_build_metadata.get("shot_recipe_id") or "")
            for scene in plan.scenes
            if scene.prompt_build_metadata.get("shot_recipe_id")
        }
    )
    selected_motifs = sorted(
        {
            str((scene.prompt_build_metadata.get("scene_world_contract") or {}).get("motif_id") or "")
            for scene in plan.scenes
            if (scene.prompt_build_metadata.get("scene_world_contract") or {}).get("motif_id")
        }
    )

    decisions = [
        decision_entry(
            "selected_pipeline",
            "pipeline",
            str(pipeline_id or metadata.get("pipeline_id") or "unknown"),
            reason="pipeline definition selected from job metadata or default",
            inputs={"job_metadata": dict(job.metadata or {})},
        ),
        decision_entry(
            "selected_mode",
            "creative_strategy",
            mode_id or "unknown",
            reason="creative mode selected by creative system detection",
            inputs={"idea": job.idea, "script_present": bool(job.script)},
        ),
        decision_entry(
            "selected_style",
            "creative_strategy",
            style_id or "unknown",
            reason="style selected from mode playbook",
        ),
        decision_entry(
            "selected_skill_set",
            "skill_layer",
            "loaded_with_missing_report",
            reason="pipeline and mode required skills resolved for traceability",
            outputs={
                "required_skills": skill_trace.get("required_skills", []),
                "loaded_skills": skill_trace.get("loaded_skills", []),
                "missing_skills": skill_trace.get("missing_skills", []),
            },
        ),
        decision_entry(
            "selected_motif_family",
            "beat_plan",
            ", ".join(str(item) for item in motif_families) if motif_families else "not_recorded",
            reason="mode-level motif families available for flexible beat planning",
            outputs={"selected_motifs": selected_motifs},
        ),
        decision_entry(
            "selected_shot_recipe",
            "beat_plan",
            ", ".join(selected_shot_recipes) if selected_shot_recipes else "not_recorded",
            reason="shot recipes selected by existing planner contracts",
        ),
        decision_entry(
            "backend_prompt_policy",
            "model_prompting",
            str(backend_prompt_policy or {}),
            reason="backend-specific prompt policy recorded for audit",
        ),
        decision_entry(
            "approval_gate_status",
            "approval_gates",
            "initial_checkpoint_trace",
            reason="checkpoint and approval state at decision-log write time",
            outputs=checkpoint_trace or {},
        ),
        decision_entry(
            "stop_after",
            "operator_control",
            stop_after or "not_requested",
            reason="operator stop-after metadata recorded for controlled runs",
        ),
        decision_entry(
            "selected_take",
            "selection",
            "future_work",
            reason="G5 keeps decision-log contract ready; full runtime selection append remains future work",
        ),
        decision_entry(
            "quality_decision",
            "final_quality_gate",
            "future_work",
            reason="G5 keeps decision-log contract ready; final verdict append remains future work",
        ),
    ]
    return DecisionLog(
        job_id=plan.job_id,
        pipeline_id=pipeline_id or metadata.get("pipeline_id"),
        decisions=decisions,
        metadata={"version_note": "Initial G2 decision log written after planning."},
    )


def write_decision_log(path: str | Path, decision_log: DecisionLog | dict[str, Any]) -> Path:
    payload = decision_log.model_dump(mode="json") if isinstance(decision_log, DecisionLog) else decision_log
    return write_json(Path(path), payload)
