from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from agent_core.adapters.base import StoryboardAdapter, VideoAdapter, VoiceAdapter
from agent_core.assembler import ResultAssembler
from agent_core.backend_registry import BackendRegistry, build_default_registry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import (
    ArtifactRef,
    ExecutionResult,
    ImageValidationReport,
    JobInput,
    KeyframeCandidatePlan,
    KeyframeCandidateResult,
    ProductionPlan,
    ResultSummary,
    ScenePlan,
    SelectedKeyframe,
    TakePlan,
    TakeResultRecord,
    TakeRetryRecord,
    TakeValidationReport,
)
from agent_core.state_store import StateStore
from agent_core.utils import build_job_id, mirror_media_file, read_json, stable_seed, validate_image_candidate, validate_video_take


class VideoAgent:
    def __init__(
        self,
        *,
        registry: BackendRegistry | None = None,
        state_store: StateStore | None = None,
        planner: ProductionPlanner | None = None,
        assembler: ResultAssembler | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.state_store = state_store or StateStore()
        self.planner = planner or ProductionPlanner(self.registry)
        self.assembler = assembler or ResultAssembler()

    def load_job(self, source: str | Path | dict[str, Any] | JobInput) -> JobInput:
        if isinstance(source, JobInput):
            job = source
        elif isinstance(source, (str, Path)):
            job = JobInput.model_validate(read_json(source))
        else:
            job = JobInput.model_validate(source)

        if not job.job_id:
            job.job_id = build_job_id(job.idea or job.script or "job")
        return job

    def run_job(self, source: str | Path | dict[str, Any] | JobInput, *, raise_on_error: bool = False) -> ResultSummary:
        job = self.load_job(source)
        state = self.state_store.initialize(job)
        plan: ProductionPlan | None = None
        voice_result = None
        storyboard_result = None

        try:
            self.state_store.transition(state, "validated", "Job schema validated.")
            plan = self.planner.build_plan(job)
            self.state_store.save_plan(state, plan)
            self.state_store.save_scene_plan(state, plan)
            self.state_store.transition(state, "planned", "Production plan created.")

            voice_step = next(step for step in plan.steps if step.name == "voice")
            if voice_step.enabled:
                voice_adapter = self.registry.primary("voice")
                if voice_adapter is None or not isinstance(voice_adapter, VoiceAdapter):
                    raise RuntimeError("Planner enabled voice but no voice adapter is available")
                self.state_store.start_step(state, "voice", voice_adapter.name)
                voice_result = voice_adapter.generate_voice(job, plan, self.state_store.job_dir(job.job_id or ""))
                self.state_store.finish_step(state, voice_result)
                if not voice_result.success:
                    raise RuntimeError(voice_result.error or "voice generation failed")
                self.state_store.transition(state, "voice_generated", "Voice generation completed.")
                if voice_result.duration_sec is not None:
                    revised_plan = self.planner.build_plan(job, actual_voice_duration_sec=voice_result.duration_sec)
                    if revised_plan.model_dump() != plan.model_dump():
                        plan = revised_plan
                        self.state_store.save_plan(state, plan)
                        self.state_store.append_log(
                            state.job_id,
                            "Plan updated after real voice duration became available.",
                        )

            storyboard_step = next(step for step in plan.steps if step.name == "storyboard")
            if storyboard_step.enabled:
                storyboard_adapter = self.registry.primary("storyboard")
                if storyboard_adapter is None or not isinstance(storyboard_adapter, StoryboardAdapter):
                    raise RuntimeError("Planner enabled storyboard but no storyboard adapter is available")
                self.state_store.start_step(state, "storyboard", storyboard_adapter.name)
                storyboard_result = self._run_storyboard_step(job, plan, state, storyboard_adapter)
                self.state_store.finish_step(state, storyboard_result)
                self.state_store.save_storyboard_report(state, self._build_storyboard_report_payload(job, storyboard_result))
                if storyboard_result.success:
                    self._attach_storyboard_selection_to_plan(plan, storyboard_result)
                    self.state_store.append_log(state.job_id, "Storyboard generation completed.")
                else:
                    self.state_store.append_log(
                        state.job_id,
                        f"Storyboard step failed softly and video will continue without storyboard context: {storyboard_result.error}",
                    )

            video_adapter = self.registry.primary("video")
            if video_adapter is None or not isinstance(video_adapter, VideoAdapter):
                raise RuntimeError("No video adapter is available for Phase 1")

            self.state_store.start_step(state, "video", video_adapter.name)
            video_result = self._run_video_step(job, plan, state, video_adapter, voice_result)
            self.state_store.finish_step(state, video_result)
            if "scene_outputs" in video_result.metadata:
                self.state_store.save_take_report(state, self._build_take_report_payload(job, video_result))
            if not video_result.success:
                raise RuntimeError(video_result.error or "video generation failed")
            self.state_store.transition(state, "video_generated", "Video generation completed.")

            result = self.assembler.assemble(
                job,
                plan,
                state,
                self.state_store.job_dir(job.job_id or ""),
                voice_result,
                video_result,
                storyboard_result,
            )
            self.state_store.transition(state, "assembled", "Result assembled.")
            self.state_store.save_result(state, result)
            self.state_store.transition(state, "done", "Job finished successfully.")
            self.state_store.save_result(state, result)
            return result
        except Exception as exc:
            self.state_store.append_log(state.job_id, traceback.format_exc())
            self.state_store.fail(state, str(exc))
            failed_result = self.assembler.failure(
                job,
                plan,
                state,
                str(exc),
                voice_result=voice_result,
                storyboard_result=storyboard_result,
            )
            self.state_store.save_result(state, failed_result)
            if raise_on_error:
                raise
            return failed_result

    def _run_storyboard_step(
        self,
        job: JobInput,
        plan: ProductionPlan,
        state,
        storyboard_adapter: StoryboardAdapter,
    ) -> ExecutionResult:
        workspace = self.state_store.job_dir(job.job_id or "")
        selection_mode = str(plan.metadata.get("storyboard_selection_mode", "preferred_variation_then_first_valid"))
        scene_storyboards: list[dict[str, Any]] = []
        selected_scene_storyboards: list[dict[str, Any]] = []
        storyboard_artifacts: list[ArtifactRef] = []
        total_candidate_count = 0
        enabled_scene_count = 0
        failed_scene_ids: list[str] = []

        for scene in plan.scenes:
            config = scene.storyboard_config
            planned_candidates = [candidate.model_copy(deep=True) for candidate in scene.keyframe_candidates]
            if not config or not config.enabled or not planned_candidates:
                scene_storyboards.append(
                    {
                        "scene_id": scene.scene_id,
                        "scene_index": scene.index,
                        "title": scene.title,
                        "storyboard_config": config.model_dump(mode="json") if config else None,
                        "keyframe_candidates": [candidate.model_dump(mode="json") for candidate in planned_candidates],
                        "generated_candidates": [],
                        "selected_keyframe": None,
                        "selection": {
                            "selection_mode": selection_mode,
                            "technical_status": "storyboard_disabled",
                            "selection_reason": "storyboard not enabled for this scene",
                        },
                    }
                )
                continue

            enabled_scene_count += 1
            scene_workspace = workspace / "scenes" / scene.scene_id / "storyboard"
            scene_workspace.mkdir(parents=True, exist_ok=True)
            candidate_records: list[KeyframeCandidateResult] = []

            for candidate in planned_candidates:
                total_candidate_count += 1
                self.state_store.append_log(state.job_id, f"storyboard candidate {candidate.candidate_id} started")
                candidate_job = self._build_storyboard_job(job, scene, candidate)
                candidate_workspace = scene_workspace / "candidates" / candidate.candidate_id
                candidate_plan = self.planner.build_storyboard_render_plan(plan, scene, candidate)
                try:
                    candidate_result = storyboard_adapter.generate_storyboard(candidate_job, candidate_plan, candidate_workspace)
                except Exception as exc:
                    candidate_result = ExecutionResult(
                        step_name="storyboard",
                        success=False,
                        status="failed",
                        backend_name=storyboard_adapter.name,
                        backend_job_id=f"{candidate_job.job_id}_storyboard",
                        error=str(exc),
                    )

                mirrored_output_path = None
                if candidate_result.output_path and Path(candidate_result.output_path).exists():
                    source_path = Path(candidate_result.output_path)
                    mirrored_path = scene_workspace / f"{candidate.candidate_id}{source_path.suffix or '.png'}"
                    mirrored_output_path = str(mirror_media_file(source_path, mirrored_path))
                    storyboard_artifacts.append(
                        ArtifactRef(
                            key=f"{scene.scene_id}_{candidate.candidate_id}_keyframe",
                            kind="image",
                            path=mirrored_output_path,
                            origin=storyboard_adapter.name,
                            exists=Path(mirrored_output_path).exists(),
                            metadata={
                                "scene_id": scene.scene_id,
                                "scene_index": scene.index,
                                "candidate_id": candidate.candidate_id,
                                "candidate_index": candidate.candidate_index,
                                "variation_id": candidate.variation_id,
                                "variation_index": candidate.variation_index,
                                "shot_type": candidate.shot_type,
                            },
                        )
                    )

                validation = self._validate_keyframe_output(plan, candidate, mirrored_output_path or candidate_result.output_path)
                review_status = self._resolve_keyframe_review_status(candidate_result, validation)
                candidate_records.append(
                    KeyframeCandidateResult(
                        candidate_id=candidate.candidate_id,
                        scene_id=scene.scene_id,
                        candidate_index=candidate.candidate_index,
                        variation_id=candidate.variation_id,
                        variation_index=candidate.variation_index,
                        shot_type=candidate.shot_type,
                        status=candidate_result.status,
                        review_status=review_status,
                        output_path=mirrored_output_path or candidate_result.output_path,
                        output_url=candidate_result.output_url,
                        selected=False,
                        validation=validation,
                        metadata={
                            "prompt_text": candidate.prompt_text,
                            "priority_rank": candidate.priority_rank,
                            "relation_type": candidate.relation_type,
                            "backend_metadata": candidate_result.metadata,
                        },
                        error=candidate_result.error,
                    )
                )
                self.state_store.append_log(state.job_id, f"storyboard candidate {candidate.candidate_id} finished with status={candidate_result.status}")

            selected_keyframe, selection_details = self._select_keyframe_candidate(scene, candidate_records)
            if selected_keyframe is None:
                failed_scene_ids.append(scene.scene_id)
            else:
                scene.selected_keyframe = selected_keyframe
                selected_scene_storyboards.append(
                    {
                        "scene_id": scene.scene_id,
                        "scene_index": scene.index,
                        "selected_keyframe": selected_keyframe.model_dump(mode="json"),
                        "selected_keyframe_id": selected_keyframe.candidate_id,
                        "selected_variation_id": selected_keyframe.variation_id,
                        "selected_by_rule": selected_keyframe.selected_by_rule,
                        "selection_reason": selected_keyframe.selection_reason,
                    }
                )
                for record in candidate_records:
                    if record.candidate_id == selected_keyframe.candidate_id:
                        record.selected = True
                        record.review_status = "selected"
                        record.metadata["selection"] = selection_details
                        break
                if selected_keyframe.output_path:
                    storyboard_artifacts.append(
                        ArtifactRef(
                            key=f"{scene.scene_id}_selected_keyframe",
                            kind="image",
                            path=selected_keyframe.output_path,
                            origin=storyboard_adapter.name,
                            exists=Path(selected_keyframe.output_path).exists(),
                            metadata={
                                "scene_id": scene.scene_id,
                                "scene_index": scene.index,
                                "candidate_id": selected_keyframe.candidate_id,
                                "variation_id": selected_keyframe.variation_id,
                            },
                        )
                    )

            scene_storyboards.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_index": scene.index,
                    "title": scene.title,
                    "storyboard_config": config.model_dump(mode="json"),
                    "keyframe_candidates": [candidate.model_dump(mode="json") for candidate in planned_candidates],
                    "generated_candidates": [record.model_dump(mode="json") for record in candidate_records],
                    "selected_keyframe": selected_keyframe.model_dump(mode="json") if selected_keyframe else None,
                    "selection": selection_details,
                }
            )

        success = enabled_scene_count == 0 or len(failed_scene_ids) < enabled_scene_count
        error = None if success else "no storyboard keyframe selected for any enabled scene"
        return ExecutionResult(
            step_name="storyboard",
            success=success,
            status="succeeded" if success else "failed",
            backend_name=storyboard_adapter.name,
            backend_job_id=f"{job.job_id}_storyboard_batch",
            artifacts=storyboard_artifacts,
            metadata={
                "scene_count": len(plan.scenes),
                "enabled_scene_count": enabled_scene_count,
                "candidate_count": total_candidate_count,
                "selection_mode": selection_mode,
                "scene_storyboards": scene_storyboards,
                "selected_scene_storyboards": selected_scene_storyboards,
                "failed_scene_ids": failed_scene_ids,
            },
            error=error,
        )

    def _run_video_step(
        self,
        job: JobInput,
        plan: ProductionPlan,
        state,
        video_adapter: VideoAdapter,
        voice_result: ExecutionResult | None,
    ) -> ExecutionResult:
        workspace = self.state_store.job_dir(job.job_id or "")
        planned_takes_per_scene = max((len(scene.takes) for scene in plan.scenes), default=1)
        planned_variations_per_scene = max((len(scene.variations) for scene in plan.scenes), default=1)
        storyboard_enabled = bool(plan.metadata.get("storyboard_enabled", False))
        has_render_mode_context = any(
            scene.render_mode != "text_only" or scene.video_mode in {"storyboard_reference", "keyframe_conditioned"}
            for scene in plan.scenes
        )
        if len(plan.scenes) <= 1 and planned_takes_per_scene <= 1 and not storyboard_enabled and not has_render_mode_context:
            return video_adapter.generate_video(
                job,
                plan,
                workspace,
                voice_result=voice_result,
            )

        scene_outputs: list[dict[str, Any]] = []
        scene_artifacts: list[ArtifactRef] = []
        aggregate_duration_sec = 0.0
        selected_scene_outputs: list[dict[str, Any]] = []
        selection_mode = str(plan.metadata.get("selection_mode", "quality_guarded_best_valid_take"))
        creative_selection_mode = str(
            plan.metadata.get("creative_selection_mode", "rule_based_scene_variation_heuristic")
        )
        fallback_selection_mode = str(plan.metadata.get("fallback_selection_mode", "first_successful_take"))
        max_quality_retries_per_scene = int(plan.metadata.get("max_quality_retries_per_scene", 1) or 0)
        total_retry_count = 0
        previous_selected_shot_type: str | None = None
        render_mode_counts = {"text_only": 0, "storyboard_reference": 0, "keyframe_conditioned": 0}
        fallback_reasons: list[dict[str, Any]] = []

        for scene in plan.scenes:
            self.state_store.append_log(state.job_id, f"scene {scene.scene_id} started")
            scene_workspace = workspace / "scenes" / scene.scene_id
            scene_workspace.mkdir(parents=True, exist_ok=True)
            take_records: list[TakeResultRecord] = []
            retry_history: list[TakeRetryRecord] = []
            pending_takes: list[TakePlan] = [take.model_copy(deep=True) for take in scene.takes]
            next_take_index = max((take.take_index for take in pending_takes), default=0) + 1

            while pending_takes:
                take = pending_takes.pop(0)
                self.state_store.append_log(state.job_id, f"take {take.take_id} started")
                take_job = self._build_take_job(job, scene, take)
                take_workspace = scene_workspace / "takes" / take.take_id
                take_plan = self.planner.build_take_render_plan(plan, scene, take)
                take_video_step = next(step for step in take_plan.steps if step.name == "video")
                take_render_mode = str(take_video_step.params.get("render_mode", take.render_mode))
                take_video_mode = str(take_video_step.params.get("video_mode", take.video_mode))
                take_fallback_strategy = str(take_video_step.params.get("fallback_strategy", take.fallback_strategy))
                take_fallback_reason = take_video_step.params.get("fallback_reason")
                selected_keyframe_usage = dict(take_video_step.params.get("selected_keyframe_usage") or {})
                take_result = video_adapter.generate_video(take_job, take_plan, take_workspace, voice_result=None)
                mirrored_output_path = None
                if take_result.output_path and Path(take_result.output_path).exists():
                    source_path = Path(take_result.output_path)
                    mirrored_path = scene_workspace / "takes" / f"{take.take_id}{source_path.suffix or '.mp4'}"
                    mirrored_output_path = str(mirror_media_file(source_path, mirrored_path))
                    scene_artifacts.append(
                        ArtifactRef(
                            key=f"{scene.scene_id}_{take.take_id}_video",
                            kind="video",
                            path=mirrored_output_path,
                            origin=video_adapter.name,
                            exists=Path(mirrored_output_path).exists(),
                            metadata={
                                "scene_id": scene.scene_id,
                                "scene_index": scene.index,
                                "take_id": take.take_id,
                                "take_index": take.take_index,
                                "variation_id": take.variation_id,
                                "variation_index": take.variation_index,
                                "seed": take.seed,
                                "source_output_path": take_result.output_path,
                            },
                        )
                    )
                validation = self._validate_take_output(plan, scene, take, mirrored_output_path or take_result.output_path)
                attempt_number = int(take.render_params.get("retry_index", 0) or 0) + 1
                review_status = self._resolve_review_status(take_result, validation)
                take_record = TakeResultRecord(
                    take_id=take.take_id,
                    scene_id=scene.scene_id,
                    take_index=take.take_index,
                    variation_id=take.variation_id,
                    variation_index=take.variation_index,
                    shot_type=take.shot_type,
                    camera_style=take.camera_style,
                    camera_motion=take.camera_motion,
                    framing_hint=take.framing_hint,
                    prompt_variant_text=take.prompt_variant_text,
                    style_bias=take.style_bias,
                    seed=take.seed,
                    video_mode=take_video_mode,
                    render_mode=take_render_mode,
                    fallback_strategy=take_fallback_strategy,
                    fallback_reason=str(take_fallback_reason) if take_fallback_reason else None,
                    status=take_result.status,
                    review_status=review_status,
                    output_path=mirrored_output_path or take_result.output_path,
                    output_url=take_result.output_url,
                    duration_sec=validation.duration_sec if validation and validation.duration_sec is not None else take_result.duration_sec,
                    selected=False,
                    attempt_number=attempt_number,
                    is_retry=bool(take.render_params.get("is_retry", False)),
                    retry_of_take_id=take.render_params.get("retry_of_take_id"),
                    retry_reason=take.render_params.get("retry_reason"),
                    validation=validation,
                    metadata={
                        "prompt_text": take.prompt_text,
                        "selected_keyframe": scene.selected_keyframe.model_dump(mode="json") if scene.selected_keyframe else None,
                        "selected_keyframe_usage": selected_keyframe_usage,
                        "backend_metadata": take_result.metadata,
                        "quality_guard": validation.model_dump(mode="json") if validation else None,
                    },
                    error=take_result.error,
                )
                take_records.append(take_record)
                if take_result.success and validation and not validation.passed:
                    self.state_store.append_log(
                        state.job_id,
                        f"take {take.take_id} rejected by quality guard: {'; '.join(validation.issues)}",
                    )
                    if len(retry_history) < max_quality_retries_per_scene:
                        retry_take = self._build_retry_take(job, scene, take, next_take_index, len(retry_history) + 1)
                        retry_history.append(
                            TakeRetryRecord(
                                scene_id=scene.scene_id,
                                source_take_id=take.take_id,
                                retry_take_id=retry_take.take_id,
                                retry_index=len(retry_history) + 1,
                                seed=retry_take.seed,
                                reason="technical_quality_rejection",
                                source_variation_id=take.variation_id,
                                retry_variation_id=retry_take.variation_id,
                            )
                        )
                        pending_takes.append(retry_take)
                        next_take_index += 1
                        total_retry_count += 1
                        self.state_store.append_log(
                            state.job_id,
                            f"retry scheduled for {scene.scene_id}: {retry_take.take_id} after {take.take_id}",
                        )
                self.state_store.append_log(state.job_id, f"take {take.take_id} finished with status={take_result.status}")

            selected_take, selection_details = self._select_take_record(
                scene,
                take_records,
                total_scene_count=len(plan.scenes),
                previous_selected_shot_type=previous_selected_shot_type,
            )
            if selected_take is None:
                return ExecutionResult(
                    step_name="video",
                    success=False,
                    status="failed",
                    backend_name=video_adapter.name,
                    backend_job_id=f"{job.job_id}_{scene.scene_id}_takes",
                    artifacts=scene_artifacts,
                    metadata={
                        "segmentation_mode": "multi_scene" if len(plan.scenes) > 1 else "single_scene",
                        "scene_count": len(plan.scenes),
                        "variations_per_scene": planned_variations_per_scene,
                        "takes_per_scene": planned_takes_per_scene,
                        "takes_per_variation": plan.metadata.get("takes_per_variation", 1),
                        "selection_mode": selection_mode,
                        "creative_selection_mode": creative_selection_mode,
                        "storyboard_enabled": storyboard_enabled,
                        "fallback_selection_mode": fallback_selection_mode,
                        "quality_guard_enabled": True,
                        "creative_selection_enabled": True,
                        "max_quality_retries_per_scene": max_quality_retries_per_scene,
                        "total_retry_count": total_retry_count,
                        "scene_outputs": scene_outputs
                        + [
                            {
                                "scene_id": scene.scene_id,
                                "scene_index": scene.index,
                                "title": scene.title,
                                "prompt_text": scene.prompt_text,
                                "variation_count": len(scene.variations),
                                "variations": [variation.model_dump(mode="json") for variation in scene.variations],
                                "selected_keyframe": scene.selected_keyframe.model_dump(mode="json") if scene.selected_keyframe else None,
                                "selected_take_id": None,
                                "selected_variation_id": None,
                                "selected_variation": None,
                                "selected_take": None,
                                "retry_history": [entry.model_dump(mode="json") for entry in retry_history],
                                "selection": selection_details,
                                "takes": [record.model_dump(mode="json") for record in take_records],
                            }
                        ],
                    },
                    error=f"no technically valid take available for {scene.scene_id}",
                )

            selected_take.selected = True
            selected_take.review_status = "selected"
            selected_take.metadata["selection"] = selection_details
            aggregate_duration_sec += selected_take.duration_sec or scene.target_duration_sec
            previous_selected_shot_type = selected_take.shot_type
            render_mode_counts[selected_take.render_mode] = render_mode_counts.get(selected_take.render_mode, 0) + 1
            if selected_take.fallback_reason:
                fallback_reasons.append(
                    {
                        "scene_id": scene.scene_id,
                        "take_id": selected_take.take_id,
                        "render_mode": selected_take.render_mode,
                        "fallback_reason": selected_take.fallback_reason,
                    }
                )
            scene_output = {
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "title": scene.title,
                "output_path": selected_take.output_path,
                "output_url": selected_take.output_url,
                "duration_sec": selected_take.duration_sec or scene.target_duration_sec,
                "prompt_text": scene.prompt_text,
                "selected_take_id": selected_take.take_id,
                "selected_variation_id": selected_take.variation_id,
                "selected_variation": self._scene_variation_payload(scene, selected_take.variation_id),
                "selected_keyframe": scene.selected_keyframe.model_dump(mode="json") if scene.selected_keyframe else None,
                "video_mode": selected_take.video_mode,
                "planned_render_mode": scene.render_mode,
                "render_mode": selected_take.render_mode,
                "fallback_strategy": selected_take.fallback_strategy,
                "fallback_reason": selected_take.fallback_reason,
                "selected_keyframe_usage": selected_take.metadata.get("selected_keyframe_usage"),
                "selected_take": selected_take.model_dump(mode="json"),
                "take_count": len(take_records),
                "variation_count": len(scene.variations),
                "takes_per_variation": plan.metadata.get("takes_per_variation", 1),
                "valid_take_count": sum(1 for record in take_records if record.validation and record.validation.passed),
                "rejected_take_count": sum(1 for record in take_records if record.review_status == "rejected"),
                "failed_take_count": sum(1 for record in take_records if record.review_status == "failed"),
                "technical_selection_status": selection_details.get("technical_selection_status"),
                "creative_selection_status": selection_details.get("creative_selection_status"),
                "technical_score": selection_details.get("technical_score"),
                "creative_score": selection_details.get("creative_score"),
                "selected_by_rule": selection_details.get("selected_by_rule"),
                "selection_reason": selection_details.get("selection_reason"),
                "retry_history": [entry.model_dump(mode="json") for entry in retry_history],
                "selection": selection_details,
                "variations": [variation.model_dump(mode="json") for variation in scene.variations],
                "takes": [record.model_dump(mode="json") for record in take_records],
            }
            scene_outputs.append(scene_output)
            selected_scene_outputs.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_index": scene.index,
                    "title": scene.title,
                    "output_path": selected_take.output_path,
                    "output_url": selected_take.output_url,
                    "duration_sec": selected_take.duration_sec or scene.target_duration_sec,
                    "selected_take_id": selected_take.take_id,
                    "selected_variation_id": selected_take.variation_id,
                    "selected_keyframe": scene.selected_keyframe.model_dump(mode="json") if scene.selected_keyframe else None,
                    "video_mode": selected_take.video_mode,
                    "planned_render_mode": scene.render_mode,
                    "render_mode": selected_take.render_mode,
                    "fallback_strategy": selected_take.fallback_strategy,
                    "fallback_reason": selected_take.fallback_reason,
                    "selected_keyframe_usage": selected_take.metadata.get("selected_keyframe_usage"),
                    "review_status": selected_take.review_status,
                    "technical_score": selection_details.get("technical_score"),
                    "creative_score": selection_details.get("creative_score"),
                    "selected_by_rule": selection_details.get("selected_by_rule"),
                    "selection_reason": selection_details.get("selection_reason"),
                    "validation": selected_take.validation.model_dump(mode="json") if selected_take.validation else None,
                }
            )
            if selected_take.output_path:
                scene_artifacts.append(
                    ArtifactRef(
                        key=f"{scene.scene_id}_selected_video",
                        kind="video",
                        path=selected_take.output_path,
                        origin=video_adapter.name,
                        exists=Path(selected_take.output_path).exists(),
                        metadata={
                            "scene_id": scene.scene_id,
                            "scene_index": scene.index,
                            "title": scene.title,
                            "selected_take_id": selected_take.take_id,
                            "selected_variation_id": selected_take.variation_id,
                        },
                    )
                )
            self.state_store.append_log(
                state.job_id,
                f"scene {scene.scene_id} finished with selected_take={selected_take.take_id} selection_mode={selection_mode}",
            )

        return ExecutionResult(
            step_name="video",
            success=True,
            status="succeeded",
            backend_name=video_adapter.name,
            backend_job_id=f"{job.job_id}_video_batch",
            output_path=selected_scene_outputs[0]["output_path"] if len(selected_scene_outputs) == 1 else None,
            output_url=selected_scene_outputs[0]["output_url"] if len(selected_scene_outputs) == 1 else None,
            duration_sec=round(aggregate_duration_sec, 3),
            artifacts=scene_artifacts,
            metadata={
                "segmentation_mode": "multi_scene" if len(plan.scenes) > 1 else "single_scene",
                "scene_count": len(scene_outputs),
                "variations_per_scene": planned_variations_per_scene,
                "takes_per_scene": planned_takes_per_scene,
                "takes_per_variation": plan.metadata.get("takes_per_variation", 1),
                "video_mode_requested": plan.metadata.get("video_mode_requested", "auto"),
                "planned_render_mode": plan.metadata.get("planned_render_mode", "text_only"),
                "render_mode_counts": render_mode_counts,
                "fallback_reasons": fallback_reasons,
                "selection_mode": selection_mode,
                "creative_selection_mode": creative_selection_mode,
                "storyboard_enabled": storyboard_enabled,
                "fallback_selection_mode": fallback_selection_mode,
                "quality_guard_enabled": True,
                "creative_selection_enabled": True,
                "max_quality_retries_per_scene": max_quality_retries_per_scene,
                "total_retry_count": total_retry_count,
                "scene_outputs": scene_outputs,
                "selected_scene_outputs": selected_scene_outputs,
                "duration_contract": {
                    "planned_duration_sec": plan.target_duration_sec,
                    "scene_duration_sum_sec": round(sum(scene.target_duration_sec for scene in plan.scenes), 3),
                },
            },
        )

    def _build_take_report_payload(self, job: JobInput, video_result: ExecutionResult) -> dict[str, Any]:
        return {
            "job_id": job.job_id or "",
            "scene_count": video_result.metadata.get("scene_count", 0),
            "variations_per_scene": video_result.metadata.get("variations_per_scene", 1),
            "takes_per_scene": video_result.metadata.get("takes_per_scene", 1),
            "takes_per_variation": video_result.metadata.get("takes_per_variation", 1),
            "video_mode_requested": video_result.metadata.get("video_mode_requested", "auto"),
            "planned_render_mode": video_result.metadata.get("planned_render_mode", "text_only"),
            "render_mode_counts": video_result.metadata.get(
                "render_mode_counts",
                {"text_only": 0, "storyboard_reference": 0, "keyframe_conditioned": 0},
            ),
            "fallback_reasons": video_result.metadata.get("fallback_reasons", []),
            "selection_mode": video_result.metadata.get("selection_mode", "quality_guarded_best_valid_take"),
            "creative_selection_mode": video_result.metadata.get(
                "creative_selection_mode", "rule_based_scene_variation_heuristic"
            ),
            "storyboard_enabled": video_result.metadata.get("storyboard_enabled", False),
            "fallback_selection_mode": video_result.metadata.get("fallback_selection_mode", "first_successful_take"),
            "quality_guard_enabled": video_result.metadata.get("quality_guard_enabled", True),
            "creative_selection_enabled": video_result.metadata.get("creative_selection_enabled", True),
            "max_quality_retries_per_scene": video_result.metadata.get("max_quality_retries_per_scene", 1),
            "total_retry_count": video_result.metadata.get("total_retry_count", 0),
            "scene_outputs": video_result.metadata.get("scene_outputs", []),
            "selected_scene_outputs": video_result.metadata.get("selected_scene_outputs", []),
        }

    def _build_take_job(self, job: JobInput, scene: ScenePlan, take: TakePlan) -> JobInput:
        metadata = dict(job.metadata)
        metadata.update(
            {
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "take_id": take.take_id,
                "take_index": take.take_index,
                "take_seed": take.seed,
                "variation_id": take.variation_id,
                "variation_index": take.variation_index,
                "shot_type": take.shot_type,
                "camera_style": take.camera_style,
                "camera_motion": take.camera_motion,
                "framing_hint": take.framing_hint,
                "style_bias": take.style_bias,
                "video_mode": take.video_mode,
                "render_mode": take.render_mode,
                "fallback_strategy": take.fallback_strategy,
                "selected_keyframe_candidate_id": scene.selected_keyframe.candidate_id if scene.selected_keyframe else None,
                "selected_keyframe_variation_id": scene.selected_keyframe.variation_id if scene.selected_keyframe else None,
                "selected_keyframe_path": scene.selected_keyframe.output_path if scene.selected_keyframe else None,
                "is_retry_take": bool(take.render_params.get("is_retry", False)),
                "retry_of_take_id": take.render_params.get("retry_of_take_id"),
                "retry_index": int(take.render_params.get("retry_index", 0) or 0),
            }
        )
        return job.model_copy(update={"job_id": f"{job.job_id}_{take.take_id}", "metadata": metadata})

    def _build_storyboard_report_payload(self, job: JobInput, storyboard_result: ExecutionResult) -> dict[str, Any]:
        return {
            "job_id": job.job_id or "",
            "scene_count": storyboard_result.metadata.get("scene_count", 0),
            "enabled_scene_count": storyboard_result.metadata.get("enabled_scene_count", 0),
            "candidate_count": storyboard_result.metadata.get("candidate_count", 0),
            "selection_mode": storyboard_result.metadata.get("selection_mode", "preferred_variation_then_first_valid"),
            "scene_storyboards": storyboard_result.metadata.get("scene_storyboards", []),
            "selected_scene_storyboards": storyboard_result.metadata.get("selected_scene_storyboards", []),
            "failed_scene_ids": storyboard_result.metadata.get("failed_scene_ids", []),
        }

    def _attach_storyboard_selection_to_plan(self, plan: ProductionPlan, storyboard_result: ExecutionResult) -> None:
        selection_lookup = {
            entry["scene_id"]: entry.get("selected_keyframe")
            for entry in storyboard_result.metadata.get("selected_scene_storyboards", [])
            if entry.get("scene_id")
        }
        for scene in plan.scenes:
            payload = selection_lookup.get(scene.scene_id)
            if payload:
                scene.selected_keyframe = SelectedKeyframe.model_validate(payload)

    def _build_storyboard_job(self, job: JobInput, scene: ScenePlan, candidate: KeyframeCandidatePlan) -> JobInput:
        metadata = dict(job.metadata)
        metadata.update(
            {
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "variation_id": candidate.variation_id,
                "variation_index": candidate.variation_index,
                "shot_type": candidate.shot_type,
                "storyboard_prompt": candidate.prompt_text,
            }
        )
        return job.model_copy(update={"job_id": f"{job.job_id}_{candidate.candidate_id}", "metadata": metadata})

    def _build_retry_take(
        self,
        job: JobInput,
        scene: ScenePlan,
        source_take: TakePlan,
        next_take_index: int,
        retry_index: int,
    ) -> TakePlan:
        if len(scene.variations) > 1 and source_take.variation_id:
            take_id = f"{source_take.variation_id}_retry_{retry_index:02d}_take_{next_take_index:02d}"
        else:
            take_id = f"{scene.scene_id}_take_{next_take_index:02d}_retry_{retry_index:02d}"
        seed = stable_seed(
            f"{job.job_id or job.primary_text}:{scene.scene_id}:{source_take.take_id}:retry:{retry_index}:{next_take_index}"
        )
        render_params = dict(source_take.render_params)
        render_params.update(
            {
                "seed": seed,
                "take_id": take_id,
                "take_index": next_take_index,
                "is_retry": True,
                "retry_of_take_id": source_take.take_id,
                "retry_index": retry_index,
                "retry_reason": "technical_quality_rejection",
            }
        )
        notes = list(source_take.notes)
        notes.append("Phase 2C retry take generated after a technical quality rejection.")
        return TakePlan(
            take_id=take_id,
            scene_id=scene.scene_id,
            take_index=next_take_index,
            variation_id=source_take.variation_id,
            variation_index=source_take.variation_index,
            shot_type=source_take.shot_type,
            camera_style=source_take.camera_style,
            camera_motion=source_take.camera_motion,
            framing_hint=source_take.framing_hint,
            prompt_variant_text=source_take.prompt_variant_text,
            style_bias=source_take.style_bias,
            seed=seed,
            prompt_text=source_take.prompt_text,
            render_params=render_params,
            notes=notes,
        )

    def _validate_take_output(
        self,
        plan: ProductionPlan,
        scene: ScenePlan,
        take: TakePlan,
        output_path: str | None,
    ) -> TakeValidationReport | None:
        if not output_path:
            return None
        expected_frame_rate = float(take.render_params.get("frame_rate", scene.render_params.get("frame_rate", plan.metadata.get("frame_rate", 24))))
        expected_duration_sec = float(
            take.render_params.get("planned_duration_sec", scene.target_duration_sec)
        )
        payload = validate_video_take(
            output_path,
            expected_width=plan.width,
            expected_height=plan.height,
            expected_frame_rate=expected_frame_rate,
            expected_duration_sec=expected_duration_sec,
        )
        return TakeValidationReport.model_validate(payload)

    def _resolve_review_status(
        self,
        take_result: ExecutionResult,
        validation: TakeValidationReport | None,
    ) -> str:
        if not take_result.success:
            return "failed"
        if validation is None:
            return "failed"
        if validation.passed:
            return "passed"
        return "rejected"

    def _validate_keyframe_output(
        self,
        plan: ProductionPlan,
        candidate: KeyframeCandidatePlan,
        output_path: str | None,
    ) -> ImageValidationReport | None:
        if not output_path:
            return None
        payload = validate_image_candidate(
            output_path,
            expected_width=int(candidate.render_params.get("width", candidate.width) or candidate.width),
            expected_height=int(candidate.render_params.get("height", candidate.height) or candidate.height),
        )
        return ImageValidationReport.model_validate(payload)

    def _resolve_keyframe_review_status(
        self,
        keyframe_result: ExecutionResult,
        validation: ImageValidationReport | None,
    ) -> str:
        if not keyframe_result.success:
            return "failed"
        if validation is None:
            return "failed"
        if validation.passed:
            return "passed"
        return "rejected"

    def _select_keyframe_candidate(
        self,
        scene: ScenePlan,
        candidate_records: list[KeyframeCandidateResult],
    ) -> tuple[SelectedKeyframe | None, dict[str, Any]]:
        valid_candidates = [
            record
            for record in candidate_records
            if record.status == "succeeded" and record.output_path and record.validation and record.validation.passed
        ]
        preferred_variation_id = scene.storyboard_config.preferred_variation_id if scene.storyboard_config else None
        selection_details: dict[str, Any] = {
            "selection_mode": "preferred_variation_then_first_valid",
            "preferred_variation_id": preferred_variation_id,
            "candidate_ids": [record.candidate_id for record in valid_candidates],
            "technical_status": "valid_candidates_available",
            "selection_reason": "prefer successful valid keyframes that match the preferred scene variation",
        }
        if not valid_candidates:
            selection_details["technical_status"] = "no_valid_candidates"
            selection_details["selection_reason"] = "no technically valid storyboard candidates available"
            return None, selection_details

        preferred_candidates = [
            record for record in valid_candidates if preferred_variation_id and record.variation_id == preferred_variation_id
        ]
        candidate_pool = preferred_candidates or valid_candidates
        selected_record = min(
            candidate_pool,
            key=lambda record: (
                int((record.metadata.get("priority_rank") or record.candidate_index)),
                record.candidate_index,
            ),
        )

        if preferred_candidates:
            selected_by_rule = "preferred_variation_match"
            selection_reason = "selected the first valid keyframe candidate from the preferred storyboard variation"
            technical_status = "preferred_variation_valid"
        else:
            selected_by_rule = "priority_rank_first_valid"
            selection_reason = "preferred variation was unavailable; selected the first valid storyboard candidate by priority rank"
            technical_status = "preferred_variation_unavailable"

        selection_details.update(
            {
                "technical_status": technical_status,
                "selected_by_rule": selected_by_rule,
                "selection_reason": selection_reason,
                "selected_candidate_id": selected_record.candidate_id,
                "selected_variation_id": selected_record.variation_id,
            }
        )
        selected_keyframe = SelectedKeyframe(
            candidate_id=selected_record.candidate_id,
            scene_id=scene.scene_id,
            candidate_index=selected_record.candidate_index,
            variation_id=selected_record.variation_id,
            variation_index=selected_record.variation_index,
            shot_type=selected_record.shot_type,
            output_path=selected_record.output_path,
            output_url=selected_record.output_url,
            selected_by_rule=selected_by_rule,
            selection_reason=selection_reason,
            technical_status=technical_status,
            validation=selected_record.validation,
            metadata=dict(selected_record.metadata),
        )
        return selected_keyframe, selection_details

    def _select_take_record(
        self,
        scene: ScenePlan,
        take_records: list[TakeResultRecord],
        *,
        total_scene_count: int,
        previous_selected_shot_type: str | None,
    ) -> tuple[TakeResultRecord | None, dict[str, Any]]:
        valid_candidates = [
            record
            for record in take_records
            if record.status == "succeeded" and record.output_path and record.validation and record.validation.passed
        ]
        selection_details: dict[str, Any] = {
            "selection_mode": "quality_guarded_best_valid_take",
            "creative_selection_mode": "rule_based_scene_variation_heuristic",
            "fallback_selection_mode": "first_successful_take",
            "fallback_used": False,
            "technical_selection_status": "valid_candidates_available",
            "creative_selection_status": "pending",
            "candidate_take_ids": [record.take_id for record in valid_candidates],
            "candidate_variation_ids": sorted({record.variation_id for record in valid_candidates if record.variation_id}),
            "selection_reason": "prefer technically valid takes with the smallest duration delta and low retry cost",
        }
        if not valid_candidates:
            selection_details["technical_selection_status"] = "no_valid_candidates"
            selection_details["creative_selection_status"] = "skipped"
            selection_details["selection_reason"] = "no technically valid takes available"
            return None, selection_details

        def score(record: TakeResultRecord) -> tuple[float, int]:
            duration_delta = (
                abs(record.validation.duration_delta_sec)
                if record.validation and record.validation.duration_delta_sec is not None
                else float("inf")
            )
            retry_penalty = 1 if record.is_retry else 0
            return (duration_delta, retry_penalty)

        scene_position = self._scene_position_label(scene.index, total_scene_count)
        scored_candidates: list[dict[str, Any]] = []
        for record in valid_candidates:
            technical_score = self._compute_technical_score(record)
            creative = self._compute_creative_score(
                scene,
                record,
                scene_position=scene_position,
                previous_selected_shot_type=previous_selected_shot_type,
            )
            record.metadata["selection_scores"] = {
                "technical_score": technical_score,
                "creative_score": creative["creative_score"],
                "selected_by_rule": creative["selected_by_rule"],
                "rule_hits": creative["rule_hits"],
                "scene_position": scene_position,
            }
            scored_candidates.append(
                {
                    "take_id": record.take_id,
                    "variation_id": record.variation_id,
                    "shot_type": record.shot_type,
                    "technical_score": technical_score,
                    "creative_score": creative["creative_score"],
                    "selected_by_rule": creative["selected_by_rule"],
                    "rule_hits": creative["rule_hits"],
                }
            )

        best_score = min(score(record) for record in valid_candidates)
        best_candidates = [record for record in valid_candidates if score(record) == best_score]
        selection_details["technical_selection_status"] = (
            "single_best_technical_candidate" if len(best_candidates) == 1 else "technical_tie_creative_evaluation"
        )
        selection_details["best_score"] = {
            "duration_delta_sec": best_score[0],
            "retry_penalty": best_score[1],
        }
        best_candidate_ids = {record.take_id for record in best_candidates}
        technical_pool = [candidate for candidate in scored_candidates if candidate["take_id"] in best_candidate_ids]
        selection_details["scored_candidates"] = scored_candidates
        selection_details["technical_candidates"] = technical_pool

        best_creative_score = max(candidate["creative_score"] for candidate in technical_pool)
        creative_best_candidates = [
            record
            for record in best_candidates
            if (record.metadata.get("selection_scores") or {}).get("creative_score") == best_creative_score
        ]
        selected_take = min(creative_best_candidates, key=lambda record: record.take_index)
        selection_details["creative_selection_status"] = (
            "single_best_creative_candidate" if len(creative_best_candidates) == 1 else "creative_tie_first_valid_fallback"
        )
        selection_details["technical_score"] = (selected_take.metadata.get("selection_scores") or {}).get("technical_score")
        selection_details["creative_score"] = (selected_take.metadata.get("selection_scores") or {}).get("creative_score")
        selection_details["selected_by_rule"] = (selected_take.metadata.get("selection_scores") or {}).get(
            "selected_by_rule", "first_successful_take"
        )
        selection_details["rule_hits"] = (selected_take.metadata.get("selection_scores") or {}).get("rule_hits", [])
        selection_details["selected_take_id"] = selected_take.take_id
        selection_details["selected_variation_id"] = selected_take.variation_id
        if len(best_candidates) > 1 and len(creative_best_candidates) == 1:
            selection_details["selection_reason"] = (
                "multiple technically equal valid takes were resolved by rule-based creative selection"
            )
        elif len(creative_best_candidates) > 1:
            selection_details["fallback_used"] = True
            selection_details["selected_by_rule"] = "first_successful_take"
            selection_details["selection_reason"] = (
                "creative and technical scores tied; fell back to first successful valid take"
            )
        else:
            selection_details["selection_reason"] = "best technically valid take also satisfied creative preference rules"
        return selected_take, selection_details

    @staticmethod
    def _scene_position_label(scene_index: int, total_scene_count: int) -> str:
        if total_scene_count <= 1:
            return "single"
        if scene_index <= 1:
            return "opening"
        if scene_index >= total_scene_count:
            return "final"
        return "middle"

    def _compute_technical_score(self, record: TakeResultRecord) -> float:
        duration_delta = (
            abs(record.validation.duration_delta_sec)
            if record.validation and record.validation.duration_delta_sec is not None
            else 1.0
        )
        retry_penalty = 10.0 if record.is_retry else 0.0
        attempt_penalty = max(0, record.attempt_number - 1) * 2.0
        return round(1000.0 - (duration_delta * 1000.0) - retry_penalty - attempt_penalty, 3)

    def _compute_creative_score(
        self,
        scene: ScenePlan,
        record: TakeResultRecord,
        *,
        scene_position: str,
        previous_selected_shot_type: str | None,
    ) -> dict[str, Any]:
        shot_type = record.shot_type or "unknown"
        framing_hint = (record.framing_hint or "").lower()
        prompt_variant_text = (record.prompt_variant_text or record.metadata.get("prompt_text") or "").lower()
        scene_text = " ".join(
            part
            for part in [scene.description, scene.prompt_text, scene.narration_text or ""]
            if part
        ).lower()

        rule_hits: list[dict[str, Any]] = []

        def hit(points: int, rule: str, reason: str) -> None:
            rule_hits.append({"points": points, "rule": rule, "reason": reason})

        if scene_position in {"single", "opening"}:
            if shot_type == "establishing":
                hit(4, "opening_prefers_establishing", "opening scenes favor establishing coverage")
            elif shot_type == "hero_tableau":
                hit(3, "opening_prefers_hero", "opening scenes can favor clear hero framing")
            elif shot_type == "detail_closeup":
                hit(-2, "opening_avoids_detail_heaviness", "opening scenes should avoid jumping into detail too early")
        elif scene_position == "middle":
            if shot_type == "medium_action":
                hit(3, "middle_prefers_medium_action", "middle scenes benefit from readable action coverage")
            elif shot_type == "detail_closeup":
                hit(2, "middle_allows_detail", "middle scenes can support tighter detail coverage")
        elif scene_position == "final":
            if shot_type == "hero_tableau":
                hit(4, "final_prefers_hero", "final scenes favor a stronger hero resolution")
            elif shot_type == "establishing":
                hit(2, "final_allows_overview", "final scenes can resolve with a clean overview")
            elif shot_type == "detail_closeup":
                hit(-1, "final_avoids_detail_lock", "final scenes should usually resolve broader than a detail shot")

        if "wake" in scene_text or "boot" in scene_text or "open" in scene_text or "startup" in scene_text:
            if shot_type in {"establishing", "medium_action"}:
                hit(2, "scene_goal_startup_match", "startup-like scenes fit broader or active variations")
        if "final" in scene_text or "complete" in scene_text or "clean" in scene_text or "settling" in scene_text:
            if shot_type in {"hero_tableau", "establishing"}:
                hit(2, "scene_goal_resolution_match", "resolution scenes fit hero or overview variants")
        if "detail" in scene_text or "interface" in scene_text or "panel" in scene_text or "surface" in scene_text:
            if shot_type == "detail_closeup":
                hit(2, "scene_goal_detail_match", "detail-led scenes fit close-up variants")
        if "render" in scene_text or "motion" in scene_text or "progress" in scene_text:
            if shot_type == "medium_action":
                hit(2, "scene_goal_motion_match", "action-oriented scenes fit medium-motion variants")

        if shot_type == "establishing" and any(token in framing_hint for token in {"wide", "environmental", "overview"}):
            hit(1, "framing_supports_establishing", "framing hint reinforces an establishing shot")
        if shot_type == "hero_tableau" and any(token in framing_hint for token in {"balanced", "hero", "negative space"}):
            hit(1, "framing_supports_hero", "framing hint reinforces a hero tableau")
        if shot_type == "detail_closeup" and any(token in framing_hint for token in {"tight", "detail"}):
            hit(1, "framing_supports_detail", "framing hint reinforces a detail shot")
        if shot_type == "medium_action" and any(token in framing_hint for token in {"medium", "three-quarter"}):
            hit(1, "framing_supports_medium", "framing hint reinforces a medium action shot")

        if shot_type == "establishing" and any(token in prompt_variant_text for token in {"push-in", "geography", "context"}):
            hit(1, "prompt_supports_establishing", "prompt variant text supports establishing coverage")
        if shot_type == "medium_action" and any(token in prompt_variant_text for token in {"tracking", "movement", "closer"}):
            hit(1, "prompt_supports_medium_action", "prompt variant text supports medium action coverage")
        if shot_type == "hero_tableau" and any(token in prompt_variant_text for token in {"hero", "clarity", "readability"}):
            hit(1, "prompt_supports_hero", "prompt variant text supports a hero presentation")

        if previous_selected_shot_type:
            if previous_selected_shot_type == shot_type:
                hit(-3, "avoid_adjacent_repetition", "adjacent scenes should avoid repeating the same shot type")
            else:
                hit(1, "adjacent_diversity_bonus", "variation differs from the previous selected scene shot type")

        creative_score = sum(entry["points"] for entry in rule_hits)
        primary_rule = "first_successful_take"
        if rule_hits:
            primary_rule = max(rule_hits, key=lambda entry: (entry["points"], entry["rule"]))["rule"]

        return {
            "creative_score": creative_score,
            "selected_by_rule": primary_rule,
            "rule_hits": rule_hits,
        }

    @staticmethod
    def _scene_variation_payload(scene: ScenePlan, variation_id: str | None) -> dict[str, Any] | None:
        if not variation_id:
            return None
        for variation in scene.variations:
            if variation.variation_id == variation_id:
                return variation.model_dump(mode="json")
        return None
