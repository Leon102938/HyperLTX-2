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
    stage_contracts = metadata.get("stage_contracts") or {}
    creative_strategy = stage_contracts.get("creative_strategy") or {}
    beat_plan = stage_contracts.get("beat_plan") or {}
    review_plan = stage_contracts.get("review_plan") or {}
    candidate_scores = metadata.get("beat_plan_candidate_scores") or []
    selected_candidate = metadata.get("selected_beat_plan_candidate") or {}
    skill_context = skill_trace.get("skill_injection_context") or metadata.get("skill_injection_context") or {}
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
                "platform_skills": skill_context.get("platform_skills", []),
                "model_skills": skill_context.get("model_skills", []),
                "stage_skills": skill_context.get("stage_skills", []),
                "review_skills": skill_context.get("review_skills", []),
                "directing_skills": skill_context.get("directing_skills", []),
            },
        ),
        decision_entry(
            "selected_hook_pattern",
            "creative_strategy",
            str(metadata.get("selected_hook_pattern") or selected_candidate.get("hook_pattern") or creative_strategy.get("hook_pattern") or "not_recorded"),
            reason="hook pattern selected from scored G7 candidate when available, otherwise mode guidance",
        ),
        decision_entry(
            "beat_plan_candidate_selection",
            "beat_plan",
            str(selected_candidate.get("candidate_id") or "not_recorded"),
            reason="G7 generated beat plan candidates and selected the highest scoring candidate",
            outputs={
                "generated_candidate_count": len(metadata.get("beat_plan_candidates") or []),
                "candidate_scores": candidate_scores,
                "selected_candidate_id": selected_candidate.get("candidate_id"),
                "selected_hook_pattern": metadata.get("selected_hook_pattern") or selected_candidate.get("hook_pattern"),
                "selected_motif_sequence": metadata.get("selected_motif_sequence") or selected_candidate.get("motif_families"),
                "selected_shot_recipe_sequence": metadata.get("selected_shot_recipe_sequence") or selected_candidate.get("shot_recipes"),
                "selection_reason": selected_candidate.get("rationale"),
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
            "creative_strategy_decision",
            "creative_strategy",
            "contract_populated",
            reason="skills, mode, style, and platform policy produced a non-empty CreativeStrategy contract",
            outputs=creative_strategy,
        ),
        decision_entry(
            "beat_plan_decision",
            "beat_plan",
            "contract_populated",
            reason="planner scene roles, motif families, shot recipes, hook, and payoff produced BeatPlan",
            outputs=beat_plan,
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
            "review_plan_decision",
            "quality_review",
            "contract_populated",
            reason="review skills and stage policy produced creative/platform review criteria",
            outputs=review_plan,
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
    if metadata.get("feedback_actions"):
        decisions.append(
            decision_entry(
                "feedback_action_created",
                "feedback_policy",
                "suggested_actions_recorded",
                reason="G8 scaffold can store suggested FeedbackAction records without executing retries",
                outputs={"feedback_actions": metadata.get("feedback_actions")},
            )
        )
    if metadata.get("retry_plan"):
        retry_plan = metadata.get("retry_plan") or {}
        decisions.append(
            decision_entry(
                "retry_plan_created",
                "feedback_policy",
                "safe_retry_plan_recorded",
                reason="G8 safe retry plan records invalidation and approval requirements without executing render retries",
                outputs={"retry_plan": retry_plan},
            )
        )
        if retry_plan.get("blocked"):
            decisions.append(
                decision_entry(
                    "blocked_by_feedback",
                    "feedback_policy",
                    "blocked",
                    reason=str(retry_plan.get("reason") or "feedback policy requires review before continuing"),
                    outputs={"top_priority_action": retry_plan.get("top_priority_action")},
                )
            )
        if retry_plan.get("requires_human_approval"):
            decisions.append(
                decision_entry(
                    "human_review_required",
                    "feedback_policy",
                    "required",
                    reason="feedback policy requires human approval before retry execution",
                    outputs={"allowed_next_actions": retry_plan.get("allowed_next_actions") or []},
                )
            )
        for artifact in retry_plan.get("invalidated_artifacts") or []:
            decisions.append(
                decision_entry(
                    "artifact_invalidated",
                    "feedback_policy",
                    str(artifact),
                    reason="retry plan marks this artifact stale for the proposed next action",
                )
            )
    return DecisionLog(
        job_id=plan.job_id,
        pipeline_id=pipeline_id or metadata.get("pipeline_id"),
        decisions=decisions,
        metadata={"version_note": "Initial G2 decision log written after planning."},
    )


def append_feedback_decisions(decision_log: dict[str, Any], *, feedback_actions: list[dict[str, Any]], retry_plan: dict[str, Any]) -> dict[str, Any]:
    payload = dict(decision_log or {})
    decisions = list(payload.get("decisions") or [])
    decisions.append(
        decision_entry(
            "feedback_action_created",
            "feedback_policy",
            "suggested_actions_recorded",
            outputs={"feedback_actions": feedback_actions},
        ).model_dump(mode="json")
    )
    decisions.append(
        decision_entry(
            "retry_plan_created",
            "feedback_policy",
            "safe_retry_plan_recorded",
            outputs={"retry_plan": retry_plan},
        ).model_dump(mode="json")
    )
    if retry_plan.get("blocked"):
        decisions.append(
            decision_entry(
                "blocked_by_feedback",
                "feedback_policy",
                "blocked",
                reason=str(retry_plan.get("reason") or "feedback policy blocked continuation"),
                outputs={"top_priority_action": retry_plan.get("top_priority_action")},
            ).model_dump(mode="json")
        )
    if retry_plan.get("requires_human_approval"):
        decisions.append(
            decision_entry(
                "human_review_required",
                "feedback_policy",
                "required",
                outputs={"allowed_next_actions": retry_plan.get("allowed_next_actions") or []},
            ).model_dump(mode="json")
        )
    for artifact in retry_plan.get("invalidated_artifacts") or []:
        decisions.append(
            decision_entry(
                "artifact_invalidated",
                "feedback_policy",
                str(artifact),
                reason="retry plan marks this artifact stale for the proposed next action",
            ).model_dump(mode="json")
        )
    payload["decisions"] = decisions
    return payload


def write_decision_log(path: str | Path, decision_log: DecisionLog | dict[str, Any]) -> Path:
    payload = decision_log.model_dump(mode="json") if isinstance(decision_log, DecisionLog) else decision_log
    return write_json(Path(path), payload)
