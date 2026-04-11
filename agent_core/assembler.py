from __future__ import annotations

from pathlib import Path

from agent_core.schemas import ArtifactRef, ExecutionResult, JobInput, JobState, ProductionPlan, ResultSummary
from agent_core.utils import concat_video_segments, copy_media_file, mux_voice_into_video, probe_media_duration


class ResultAssembler:
    def assemble(
        self,
        job: JobInput,
        plan: ProductionPlan,
        state: JobState,
        workspace: Path,
        voice_result: ExecutionResult | None,
        video_result: ExecutionResult,
        storyboard_result: ExecutionResult | None = None,
    ) -> ResultSummary:
        success = bool(video_result.success)
        message = "Video agent run completed." if success else (video_result.error or "Video generation failed.")
        assembly_metadata: dict[str, object] = {}

        final_output_path: str | None = None
        actual_video_duration_sec: float | None = None
        actual_final_duration_sec: float | None = None
        scene_outputs = video_result.metadata.get("scene_outputs", [])
        selected_scene_outputs = video_result.metadata.get("selected_scene_outputs", scene_outputs)
        self._assert_selected_scene_outputs_are_valid(selected_scene_outputs)
        effective_video_path = video_result.output_path
        if success and len(selected_scene_outputs) > 1:
            assembled_video_path = workspace / "assembled_video.mp4"
            concat_video_segments(
                [scene_output["output_path"] for scene_output in selected_scene_outputs if scene_output.get("output_path")],
                assembled_video_path,
            )
            effective_video_path = str(assembled_video_path)
            assembled_artifact = ArtifactRef(
                key="assembled_video",
                kind="video",
                path=str(assembled_video_path),
                origin="agent_core.assembler",
                exists=assembled_video_path.exists(),
                metadata={
                    "scene_count": len(selected_scene_outputs),
                    "selected_take_ids": [
                        scene_output.get("selected_take_id")
                        for scene_output in selected_scene_outputs
                        if scene_output.get("selected_take_id")
                    ],
                },
            )
            self._upsert_state_artifact(state, assembled_artifact)

        if success and effective_video_path:
            final_path = workspace / "final.mp4"
            video_duration_sec = probe_media_duration(effective_video_path)
            actual_video_duration_sec = video_duration_sec
            has_voice_artifact = bool(
                voice_result
                and voice_result.output_path
                and Path(voice_result.output_path).exists()
            )
            audio_duration_sec = probe_media_duration(voice_result.output_path) if has_voice_artifact and voice_result else None

            if video_duration_sec is None:
                raise RuntimeError("Assembler could not probe the rendered video duration")

            if has_voice_artifact and voice_result and voice_result.output_path:
                mux_voice_into_video(
                    effective_video_path,
                    voice_result.output_path,
                    final_path,
                    duration_sec=video_duration_sec,
                )
                if audio_duration_sec is None:
                    timing_mode = "voice_present_duration_unknown"
                elif audio_duration_sec < video_duration_sec:
                    timing_mode = "voice_padded_to_video"
                elif audio_duration_sec > video_duration_sec:
                    timing_mode = "voice_trimmed_to_video"
                else:
                    timing_mode = "voice_duration_matches_video"
                final_duration_sec = probe_media_duration(final_path)

                assembly_metadata = {
                    "mode": "mux_voice_into_video",
                    "timing_mode": timing_mode,
                    "source_video_path": effective_video_path,
                    "source_audio_path": voice_result.output_path,
                    "final_output_path": str(final_path),
                    "planned_duration_sec": plan.target_duration_sec,
                    "video_duration_sec": video_duration_sec,
                    "voice_duration_sec": audio_duration_sec,
                    "final_duration_sec": final_duration_sec,
                    "video_minus_planned_sec": round(video_duration_sec - plan.target_duration_sec, 3),
                    "scene_count": len(plan.scenes),
                }
                message = "Video agent run completed with muxed final MP4."
                actual_final_duration_sec = final_duration_sec
            else:
                copy_media_file(effective_video_path, final_path)
                final_duration_sec = probe_media_duration(final_path)
                assembly_metadata = {
                    "mode": "copy_video_without_voice",
                    "timing_mode": "no_voice_artifact",
                    "source_video_path": effective_video_path,
                    "source_audio_path": voice_result.output_path if voice_result else None,
                    "final_output_path": str(final_path),
                    "planned_duration_sec": plan.target_duration_sec,
                    "video_duration_sec": video_duration_sec,
                    "voice_duration_sec": None,
                    "final_duration_sec": final_duration_sec,
                    "video_minus_planned_sec": round(video_duration_sec - plan.target_duration_sec, 3),
                    "scene_count": len(plan.scenes),
                }
                message = "Video agent run completed without voice; final MP4 mirrors the rendered video."
                actual_final_duration_sec = final_duration_sec

            final_output_path = str(final_path)
            final_artifact = ArtifactRef(
                key="final_output_mp4",
                kind="video",
                path=final_output_path,
                origin="agent_core.assembler",
                exists=final_path.exists(),
                metadata=assembly_metadata,
            )
            self._upsert_state_artifact(state, final_artifact)
            state.notes.append(f"Final MP4 assembled at {final_output_path}.")

        return ResultSummary(
            job_id=job.job_id or "",
            success=success,
            final_phase="assembled" if success else "failed",
            message=message,
            planned_duration_sec=plan.target_duration_sec,
            actual_voice_duration_sec=voice_result.duration_sec if voice_result else None,
            actual_video_duration_sec=actual_video_duration_sec,
            actual_final_duration_sec=actual_final_duration_sec,
            output_final_path=final_output_path,
            output_video_path=effective_video_path,
            output_audio_path=voice_result.output_path if voice_result else None,
            artifacts=list(state.artifacts),
            backend_runs={
                "voice": voice_result.model_dump(mode="json") if voice_result else None,
                "storyboard": storyboard_result.model_dump(mode="json") if storyboard_result else None,
                "video": video_result.model_dump(mode="json"),
            },
            metadata={
                "selected_pipeline": plan.selected_pipeline,
                "render_profile": plan.render_profile,
                "orientation": plan.orientation,
                "resolution": {"width": plan.width, "height": plan.height, "label": plan.resolution_label},
                "scene_count": len(plan.scenes),
                "storyboard_enabled": bool(plan.metadata.get("storyboard_enabled", False)),
                "storyboard_selection_mode": plan.metadata.get("storyboard_selection_mode"),
                "selected_scene_storyboards": (
                    storyboard_result.metadata.get("selected_scene_storyboards", []) if storyboard_result else []
                ),
                "selection_mode": video_result.metadata.get("selection_mode", "quality_guarded_best_valid_take"),
                "creative_selection_mode": video_result.metadata.get(
                    "creative_selection_mode", "rule_based_scene_variation_heuristic"
                ),
                "fallback_selection_mode": video_result.metadata.get("fallback_selection_mode", "first_successful_take"),
                "selected_scene_outputs": selected_scene_outputs,
                "assembly": assembly_metadata,
            },
        )

    def failure(
        self,
        job: JobInput,
        plan: ProductionPlan | None,
        state: JobState,
        message: str,
        voice_result: ExecutionResult | None = None,
        storyboard_result: ExecutionResult | None = None,
    ) -> ResultSummary:
        return ResultSummary(
            job_id=job.job_id or "",
            success=False,
            final_phase="failed",
            message=message,
            planned_duration_sec=plan.target_duration_sec if plan else None,
            actual_voice_duration_sec=voice_result.duration_sec if voice_result else None,
            actual_video_duration_sec=None,
            actual_final_duration_sec=None,
            output_final_path=None,
            output_audio_path=voice_result.output_path if voice_result else None,
            artifacts=list(state.artifacts),
            backend_runs={
                "voice": voice_result.model_dump(mode="json") if voice_result else None,
                "storyboard": storyboard_result.model_dump(mode="json") if storyboard_result else None,
            },
            metadata={"errors": list(state.errors)},
        )

    @staticmethod
    def _upsert_state_artifact(state: JobState, artifact: ArtifactRef) -> None:
        for index, existing in enumerate(state.artifacts):
            if existing.key == artifact.key:
                state.artifacts[index] = artifact
                return
        state.artifacts.append(artifact)

    @staticmethod
    def _assert_selected_scene_outputs_are_valid(selected_scene_outputs: list[dict[str, object]]) -> None:
        for scene_output in selected_scene_outputs:
            validation = scene_output.get("validation")
            review_status = scene_output.get("review_status")
            if validation is not None and isinstance(validation, dict) and validation.get("passed") is not True:
                raise RuntimeError(
                    f"Assembler received non-valid selected take for {scene_output.get('scene_id', 'unknown_scene')}"
                )
            if review_status is not None and review_status != "selected":
                raise RuntimeError(
                    f"Assembler received unselected scene output for {scene_output.get('scene_id', 'unknown_scene')}"
                )
