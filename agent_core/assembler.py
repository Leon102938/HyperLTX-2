from __future__ import annotations

from pathlib import Path

from agent_core.schemas import ArtifactRef, ExecutionResult, JobInput, JobState, ProductionPlan, ResultSummary
from agent_core.utils import (
    assemble_final_video,
    build_scene_subtitle_entries,
    concat_video_segments,
    copy_media_file,
    evaluate_final_quality_verdict,
    format_overlay_title_text,
    mux_voice_into_video,
    probe_media_duration,
    write_srt_subtitles,
)


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
        music_result: ExecutionResult | None = None,
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
            has_music_artifact = bool(
                music_result
                and music_result.output_path
                and Path(music_result.output_path).exists()
                and music_result.success
            )
            audio_duration_sec = probe_media_duration(voice_result.output_path) if has_voice_artifact and voice_result else None
            music_duration_sec = probe_media_duration(music_result.output_path) if has_music_artifact and music_result else None

            if video_duration_sec is None:
                raise RuntimeError("Assembler could not probe the rendered video duration")

            subtitle_mode = str(job.metadata.get("subtitle_mode", "off")).strip().lower()
            overlay_text = " ".join(str(job.metadata.get("overlay_text") or job.metadata.get("overlay_title") or "").split())
            subtitle_path: Path | None = None
            subtitle_entries: list[dict[str, object]] = []
            if subtitle_mode in {"sidecar", "burn"} and job.use_voice:
                subtitle_max_words = int(plan.metadata.get("subtitle_max_words", job.metadata.get("subtitle_max_words", 7)))
                subtitle_max_chars = int(plan.metadata.get("subtitle_max_chars", job.metadata.get("subtitle_max_chars", 42)))
                subtitle_min_words = int(plan.metadata.get("subtitle_min_words", job.metadata.get("subtitle_min_words", 2)))
                subtitle_min_chars = int(plan.metadata.get("subtitle_min_chars", job.metadata.get("subtitle_min_chars", 8)))
                subtitle_min_duration_sec = float(
                    plan.metadata.get(
                        "subtitle_min_duration_sec",
                        job.metadata.get("subtitle_min_duration_sec", 1.0),
                    )
                )
                subtitle_entries = build_scene_subtitle_entries(
                    plan.scenes,
                    max_words=subtitle_max_words,
                    max_chars=subtitle_max_chars,
                    min_words=subtitle_min_words,
                    min_chars=subtitle_min_chars,
                    min_segment_duration_sec=subtitle_min_duration_sec,
                )
                if subtitle_entries:
                    subtitle_path = write_srt_subtitles(workspace / "captions.srt", subtitle_entries)
                    subtitle_artifact = ArtifactRef(
                        key="subtitle_file",
                        kind="subtitle",
                        path=str(subtitle_path),
                        origin="agent_core.assembler",
                        exists=subtitle_path.exists(),
                        metadata={"subtitle_mode": subtitle_mode, "entry_count": len(subtitle_entries)},
                    )
                    self._upsert_state_artifact(state, subtitle_artifact)

            overlay_text_file: Path | None = None
            if overlay_text:
                overlay_text_file = workspace / "overlay_title.txt"
                formatted_overlay_text = format_overlay_title_text(
                    overlay_text,
                    max_chars_per_line=int(job.metadata.get("overlay_max_chars_per_line", 18)),
                    max_lines=int(job.metadata.get("overlay_max_lines", 3)),
                )
                overlay_text_file.write_text(formatted_overlay_text + "\n", encoding="utf-8")
                overlay_artifact = ArtifactRef(
                    key="overlay_title_text",
                    kind="text",
                    path=str(overlay_text_file),
                    origin="agent_core.assembler",
                    exists=overlay_text_file.exists(),
                )
                self._upsert_state_artifact(state, overlay_artifact)

            needs_enhanced_assembly = bool(
                has_music_artifact or subtitle_mode == "burn" or overlay_text or (has_voice_artifact and subtitle_mode == "sidecar")
            )

            if needs_enhanced_assembly:
                assemble_final_video(
                    effective_video_path,
                    final_path,
                    duration_sec=video_duration_sec,
                    voice_path=voice_result.output_path if has_voice_artifact and voice_result else None,
                    music_path=music_result.output_path if has_music_artifact and music_result else None,
                    subtitle_path=subtitle_path,
                    burn_subtitles=subtitle_mode == "burn" and subtitle_path is not None,
                    overlay_text_file=overlay_text_file,
                    overlay_duration_sec=float(job.metadata.get("overlay_duration_sec", 3.5)),
                    voice_volume=float(job.metadata.get("voice_volume", 1.0)),
                    music_volume=float(job.metadata.get("music_volume", 0.18 if has_voice_artifact else 0.35)),
                    music_fade_out_sec=float(job.metadata.get("music_fade_out_sec", 1.5)),
                )
                final_duration_sec = probe_media_duration(final_path)
                timing_mode = (
                    "voice_music_mixed"
                    if has_voice_artifact and has_music_artifact
                    else "music_only_mixed"
                    if has_music_artifact
                    else "voice_with_finishing"
                )
                assembly_metadata = {
                    "mode": "enhanced_mix",
                    "timing_mode": timing_mode,
                    "source_video_path": effective_video_path,
                    "source_audio_path": voice_result.output_path if voice_result else None,
                    "source_music_path": music_result.output_path if music_result else None,
                    "subtitle_file_path": str(subtitle_path) if subtitle_path else None,
                    "subtitle_mode": subtitle_mode,
                    "subtitle_burned": subtitle_mode == "burn" and subtitle_path is not None,
                    "subtitle_entry_count": len(subtitle_entries),
                    "overlay_text": overlay_text or None,
                    "final_output_path": str(final_path),
                    "planned_duration_sec": plan.target_duration_sec,
                    "video_duration_sec": video_duration_sec,
                    "voice_duration_sec": audio_duration_sec,
                    "music_duration_sec": music_duration_sec,
                    "final_duration_sec": final_duration_sec,
                    "video_minus_planned_sec": round(video_duration_sec - plan.target_duration_sec, 3),
                    "scene_count": len(plan.scenes),
                }
                message = "Video agent run completed with music/subtitle-aware final assembly."
                actual_final_duration_sec = final_duration_sec
            elif has_voice_artifact and voice_result and voice_result.output_path:
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
                    "source_music_path": None,
                    "subtitle_file_path": str(subtitle_path) if subtitle_path else None,
                    "subtitle_mode": subtitle_mode,
                    "subtitle_burned": False,
                    "subtitle_entry_count": len(subtitle_entries),
                    "overlay_text": overlay_text or None,
                    "final_output_path": str(final_path),
                    "planned_duration_sec": plan.target_duration_sec,
                    "video_duration_sec": video_duration_sec,
                    "voice_duration_sec": audio_duration_sec,
                    "music_duration_sec": music_duration_sec,
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
                    "source_music_path": music_result.output_path if music_result else None,
                    "subtitle_file_path": str(subtitle_path) if subtitle_path else None,
                    "subtitle_mode": subtitle_mode,
                    "subtitle_burned": False,
                    "subtitle_entry_count": len(subtitle_entries),
                    "overlay_text": overlay_text or None,
                    "final_output_path": str(final_path),
                    "planned_duration_sec": plan.target_duration_sec,
                    "video_duration_sec": video_duration_sec,
                    "voice_duration_sec": None,
                    "music_duration_sec": music_duration_sec,
                    "final_duration_sec": final_duration_sec,
                    "video_minus_planned_sec": round(video_duration_sec - plan.target_duration_sec, 3),
                    "scene_count": len(plan.scenes),
                }
                message = "Video agent run completed without voice; final MP4 mirrors the rendered video."
                actual_final_duration_sec = final_duration_sec

            final_output_path = str(final_path)
            final_quality_verdict = evaluate_final_quality_verdict(
                final_output_path=final_output_path,
                expected_width=plan.width,
                expected_height=plan.height,
                expected_frame_rate=float(plan.metadata.get("frame_rate", 24) or 24),
                expected_duration_sec=actual_final_duration_sec or video_duration_sec,
                selected_scene_outputs=selected_scene_outputs,
                selected_scene_storyboards=storyboard_result.metadata.get("selected_scene_storyboards", []) if storyboard_result else [],
                assembly_metadata=assembly_metadata,
                output_dir=workspace,
                final_frame_enabled=plan.metadata.get("vision_review_enabled"),
                final_frame_provider=plan.metadata.get("final_quality_review_provider") or plan.metadata.get("vision_review_provider"),
                final_frame_model_dir=plan.metadata.get("vision_review_model_dir"),
                max_final_frames=int(plan.metadata.get("final_quality_max_frames", plan.metadata.get("vision_review_max_frames", 3)) or 3),
                voice_metadata=voice_result.model_dump(mode="json") if voice_result else None,
                music_metadata=music_result.model_dump(mode="json") if music_result else None,
            )
            assembly_metadata["final_quality_verdict"] = final_quality_verdict
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
                "music": music_result.model_dump(mode="json") if music_result else None,
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
                "music_requested": bool(plan.metadata.get("music_requested", False)),
                "storyboard_selection_mode": plan.metadata.get("storyboard_selection_mode"),
                "selected_scene_storyboards": (
                    storyboard_result.metadata.get("selected_scene_storyboards", []) if storyboard_result else []
                ),
                "video_mode_requested": plan.metadata.get("video_mode_requested", "auto"),
                "planned_render_mode": plan.metadata.get("planned_render_mode", "text_only"),
                "render_mode_counts": video_result.metadata.get(
                    "render_mode_counts",
                    {"text_only": 0, "storyboard_reference": 0, "keyframe_conditioned": 0},
                ),
                "fallback_reasons": video_result.metadata.get("fallback_reasons", []),
                "director_mode": plan.metadata.get("director_mode"),
                "director_llm_active": plan.metadata.get("director_llm_active"),
                "director_fallback_reason": plan.metadata.get("director_fallback_reason"),
                "director_llm_provider": plan.metadata.get("director_llm_provider"),
                "director_llm_model": plan.metadata.get("director_llm_model"),
                "director_llm_endpoint": plan.metadata.get("director_llm_endpoint"),
                "director_output": plan.director_output.model_dump(mode="json") if plan.director_output else None,
                "style_lock": plan.metadata.get("style_lock"),
                "prompt_guidance": plan.metadata.get("prompt_guidance"),
                "selection_mode": video_result.metadata.get("selection_mode", "quality_guarded_best_valid_take"),
                "creative_selection_mode": video_result.metadata.get(
                    "creative_selection_mode", "rule_based_scene_variation_heuristic"
                ),
                "fallback_selection_mode": video_result.metadata.get("fallback_selection_mode", "first_successful_take"),
                "selected_scene_outputs": selected_scene_outputs,
                "assembly": assembly_metadata,
                "final_quality_verdict": assembly_metadata.get("final_quality_verdict"),
            },
        )

    def failure(
        self,
        job: JobInput,
        plan: ProductionPlan | None,
        state: JobState,
        message: str,
        voice_result: ExecutionResult | None = None,
        music_result: ExecutionResult | None = None,
        storyboard_result: ExecutionResult | None = None,
    ) -> ResultSummary:
        final_quality_verdict = evaluate_final_quality_verdict(
            final_output_path=None,
            expected_width=plan.width if plan else 0,
            expected_height=plan.height if plan else 0,
            expected_frame_rate=float(plan.metadata.get("frame_rate", 24) if plan else 24),
            expected_duration_sec=plan.target_duration_sec if plan else 0.0,
            selected_scene_outputs=[],
            selected_scene_storyboards=storyboard_result.metadata.get("selected_scene_storyboards", []) if storyboard_result else [],
            assembly_metadata={"mode": "failed_before_assembly", "errors": list(state.errors)},
        )
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
                "music": music_result.model_dump(mode="json") if music_result else None,
                "storyboard": storyboard_result.model_dump(mode="json") if storyboard_result else None,
            },
            metadata={"errors": list(state.errors), "final_quality_verdict": final_quality_verdict},
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
