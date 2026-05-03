from __future__ import annotations

from typing import Any

from agent_core.schemas import (
    BeatPlan,
    CreativeStrategy,
    JobInput,
    ModelPromptPlan,
    ProductionPlan,
    ReviewPlan,
    VisualDirection,
)


CONTRACT_VERSION = "g3_stage_role_contracts_v1"


def build_stage_role_contracts(
    *,
    job: JobInput,
    plan: ProductionPlan,
    mode: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    loaded_skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mode = dict(mode or {})
    style = dict(style or {})
    loaded_skill_ids = [str(skill.get("skill_id")) for skill in loaded_skills or [] if isinstance(skill, dict)]
    model_skill_ids = [skill_id for skill_id in loaded_skill_ids if skill_id.startswith("models/")]

    scene_roles = {
        scene.scene_id: (scene.scene_intent.narrative_role if scene.scene_intent else str(scene.index))
        for scene in plan.scenes
    }
    selected_shot_recipes = [
        str(scene.prompt_build_metadata.get("shot_recipe_id"))
        for scene in plan.scenes
        if scene.prompt_build_metadata.get("shot_recipe_id")
    ]
    selected_motifs = [
        str((scene.prompt_build_metadata.get("scene_world_contract") or {}).get("motif_id"))
        for scene in plan.scenes
        if (scene.prompt_build_metadata.get("scene_world_contract") or {}).get("motif_id")
    ]
    first_scene = plan.scenes[0] if plan.scenes else None
    first_meta = first_scene.prompt_build_metadata if first_scene else {}
    first_contract = dict((first_meta or {}).get("scene_world_contract") or {})

    strategy = CreativeStrategy(
        strategy_id=f"{plan.job_id}:creative_strategy",
        mode_id=str(plan.metadata.get("mode_id") or mode.get("mode_id") or ""),
        style_id=str(plan.metadata.get("style_id") or style.get("style_id") or ""),
        user_intent=job.primary_text,
        creative_goal=str(mode.get("creative_goal") or plan.prompt_text),
        platform=str(mode.get("platform_default") or plan.metadata.get("platform") or "shortform"),
        target_platform=str(mode.get("platform_default") or plan.metadata.get("platform") or "shortform"),
        hook_pattern=str((mode.get("hook_patterns") or [""])[0] or ""),
        pacing=dict(mode.get("pacing") or {}),
        creative_freedom=str(mode.get("shot_recipe_policy") or "select from safe motif families"),
        continuity_mode=str(plan.metadata.get("selection_mode") or "quality_guarded_best_valid_take"),
        audience_intent=str(mode.get("audience_feel") or ""),
        audience_feel=str(mode.get("audience_feel") or ""),
        success_criteria=list((mode.get("quality_targets") or {}).values()),
        anti_goals=list(mode.get("anti_patterns") or []),
        motif_families=list(plan.metadata.get("motif_families") or mode.get("motif_families") or []),
        constraints=list(mode.get("global_forbidden") or []),
        skill_ids=list(plan.metadata.get("required_skills") or []),
        metadata={"contract_version": CONTRACT_VERSION},
    )

    beat_plan = BeatPlan(
        beat_plan_id=f"{plan.job_id}:beat_plan",
        beats=[
            {
                "scene_id": scene.scene_id,
                "role": scene_roles.get(scene.scene_id),
                "duration_sec": scene.target_duration_sec,
                "motif_id": (scene.prompt_build_metadata.get("scene_world_contract") or {}).get("motif_id"),
                "shot_recipe_id": scene.prompt_build_metadata.get("shot_recipe_id"),
                "visual_goal": scene.scene_intent.visual_goal if scene.scene_intent else scene.description,
            }
            for scene in plan.scenes
        ],
        scene_roles=scene_roles,
        timing_intent=f"{len(plan.scenes)} scenes over {plan.target_duration_sec:.3f}s",
        escalation_logic="hook -> tactile/clarity beat -> payoff",
        payoff=plan.scenes[-1].description if plan.scenes else None,
        selected_motif_families=list(plan.metadata.get("motif_families") or []),
        selected_shot_recipes=selected_shot_recipes,
        transition_notes=[scene.scene_intent.transition_note for scene in plan.scenes if scene.scene_intent and scene.scene_intent.transition_note],
        metadata={"selected_motifs": selected_motifs, "contract_version": CONTRACT_VERSION},
    )

    visual_direction = VisualDirection(
        direction_id=f"{plan.job_id}:visual_direction",
        visual_identity=str((plan.director_output.style_lock.visual_identity if plan.director_output else style.get("visual_identity")) or ""),
        motif_family=str((plan.metadata.get("motif_families") or [""])[0] or ""),
        shot_recipe=selected_shot_recipes[0] if selected_shot_recipes else None,
        lighting=str((plan.director_output.style_lock.lighting if plan.director_output else style.get("lighting")) or first_contract.get("lighting") or ""),
        camera_language=str((plan.director_output.style_lock.camera_language if plan.director_output else style.get("camera_language")) or first_contract.get("camera") or ""),
        motion_language=str(first_contract.get("visual_energy_level") or ""),
        movement=str(first_contract.get("action") or ""),
        composition_rules=["single full-frame physical scene", "no split-screen or collage layout"],
        object_count_policy=str((mode.get("quality_targets") or {}).get("object_count") or ""),
        human_action_policy=str((mode.get("quality_targets") or {}).get("motion") or ""),
        avoid_risks=list(first_contract.get("forbidden_props") or mode.get("global_forbidden") or []),
        allowed_visuals=list(first_contract.get("allowed_props") or []),
        forbidden_visuals=list(first_contract.get("forbidden_props") or mode.get("global_forbidden") or []),
        skill_ids=[skill_id for skill_id in loaded_skill_ids if skill_id.startswith(("directing/", "stages/visual_direction"))],
        metadata={"contract_version": CONTRACT_VERSION},
    )

    model_prompt_plan = ModelPromptPlan(
        prompt_plan_id=f"{plan.job_id}:model_prompt_plan",
        backend_prompt_policy=dict(plan.metadata.get("backend_prompt_policy") or {}),
        positive_model_prompt=str(first_meta.get("positive_model_prompt") or first_contract.get("positive_model_prompt") or ""),
        negative_model_prompt=str(first_meta.get("negative_model_prompt") or first_contract.get("negative_model_prompt") or ""),
        zimage_prompt_sent=str(first_meta.get("zimage_prompt_sent") or first_contract.get("zimage_prompt_sent") or ""),
        ltx_positive_prompt_sent=str(first_meta.get("positive_model_prompt") or first_contract.get("positive_model_prompt") or ""),
        ltx_negative_prompt_sent=str(first_meta.get("negative_model_prompt") or first_contract.get("negative_model_prompt") or ""),
        warnings=[],
        skill_ids=[skill_id for skill_id in loaded_skill_ids if skill_id.startswith(("prompting/", "models/"))],
        loaded_model_skills=model_skill_ids,
        metadata={"contract_version": CONTRACT_VERSION, "ltx_negative_prompt_supported": False},
    )

    review_plan = ReviewPlan(
        review_plan_id=f"{plan.job_id}:review_plan",
        provider=str(plan.metadata.get("vision_review_provider") or "heuristic"),
        technical_checks=["file_exists", "decode_ok", "duration_match", "resolution_match"],
        checks=["technical_validity", "artifact_detection", "creative_quality", "platform_fit"],
        creative_quality_checks=[
            "boring_scene",
            "weak_hook",
            "unclear_action",
            "generic_stock_feel",
            "physical_incoherence",
            "bad_composition",
            "no_visual_change",
            "dead_static_scene",
            "confusing_subject",
            "voice_visual_mismatch",
        ],
        platform_fit_checks=["portrait_readability", "first_beat_hook", "mobile_composition"],
        artifact_checks=["visible_text", "phone_or_ui", "split_screen", "collage", "labels_or_logos"],
        rejection_rules=["visible text/UI/device in clean lifestyle mode", "technical validation failed", "off-topic motif"],
        selection_policy=str(plan.metadata.get("selection_mode") or "quality_guarded_best_valid_take"),
        skill_ids=[skill_id for skill_id in loaded_skill_ids if skill_id.startswith(("review/", "stages/quality_review", "models/qwen3_vl_review"))],
        metadata={"contract_version": CONTRACT_VERSION},
    )

    return {
        "contract_version": CONTRACT_VERSION,
        "creative_strategy": strategy.model_dump(mode="json"),
        "beat_plan": beat_plan.model_dump(mode="json"),
        "visual_direction": visual_direction.model_dump(mode="json"),
        "model_prompt_plan": model_prompt_plan.model_dump(mode="json"),
        "review_plan": review_plan.model_dump(mode="json"),
    }
