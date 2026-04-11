from __future__ import annotations

import math
import re
from typing import Any

from agent_core.backend_registry import BackendRegistry
from agent_core.schemas import (
    JobInput,
    KeyframeCandidatePlan,
    ProductionPlan,
    ProductionStep,
    ScenePlan,
    ShotPlan,
    StoryboardConfig,
    TakePlan,
    VariationPlan,
)
from agent_core.utils import choose_resolution, estimate_speech_duration, quantize_duration_to_frame_contract, stable_seed


class ProductionPlanner:
    MIN_VIDEO_DURATION_SEC = 4.0
    MIN_SCENE_DURATION_SEC = 2.0
    MAX_SCENES = 6
    MAX_VARIATIONS_PER_SCENE = 4
    MAX_TAKES_PER_SCENE = 4
    MAX_TAKE_RETRIES_PER_SCENE = 2
    MAX_STORYBOARD_CANDIDATES = 4
    VOICE_PADDING_SEC = 1.0
    DEFAULT_FRAME_RATE = 24

    def __init__(self, registry: BackendRegistry) -> None:
        self.registry = registry

    def build_plan(self, job: JobInput, actual_voice_duration_sec: float | None = None) -> ProductionPlan:
        video_capability = self.registry.primary_capability("video")
        if video_capability is None:
            raise ValueError("No phase-1 video backend is available")

        voice_capability = self.registry.primary_capability("voice")
        storyboard_capability = self.registry.primary_capability("storyboard")
        music_capability = self.registry.primary_capability("music")

        warnings: list[str] = []
        rules_applied: list[str] = []

        if job.use_voice and voice_capability is None:
            raise ValueError("Job requests voice, but no phase-1 voice backend is available")

        render_profile = self._resolve_render_profile(job.pipeline_preference)
        selected_pipeline = self._resolve_pipeline(job, job.use_voice)
        if not self.registry.supports_video_pipeline(selected_pipeline):
            raise ValueError(f"Video backend does not support planned pipeline '{selected_pipeline}'")

        width, height, resolution_label = choose_resolution(job.orientation or "landscape", job.resolution)
        requested_duration = round(job.duration_sec, 2) if job.duration_sec else None

        estimated_voice_duration = None
        target_duration = max(requested_duration or 0.0, self.MIN_VIDEO_DURATION_SEC)
        if job.use_voice:
            estimated_voice_duration = (
                round(actual_voice_duration_sec, 2)
                if actual_voice_duration_sec is not None
                else estimate_speech_duration(job.script or job.idea)
            )
            target_duration = max(target_duration, estimated_voice_duration + self.VOICE_PADDING_SEC)
            if actual_voice_duration_sec is not None:
                rules_applied.append(
                    "Actual voice duration was used to resize the final video length with 1s guard padding."
                )
            else:
                rules_applied.append(
                    "Estimated voice duration was used to size the final video length with 1s guard padding."
                )
        else:
            rules_applied.append("Voice disabled; final video duration follows requested duration or minimum fallback.")

        snapped_num_frames, snapped_duration = quantize_duration_to_frame_contract(
            target_duration,
            self.DEFAULT_FRAME_RATE,
        )
        if snapped_duration != round(target_duration, 3):
            rules_applied.append(
                "Video duration was snapped to the LTX frame contract (8k+1 frames) for backend stability."
            )
        target_duration = snapped_duration

        scenes = self._build_scene_plans(
            job,
            width=width,
            height=height,
            resolution_label=resolution_label,
            render_profile=render_profile,
            selected_pipeline=selected_pipeline,
            target_duration_sec=target_duration,
            frame_rate=self.DEFAULT_FRAME_RATE,
        )
        scene_total_duration = round(sum(scene.target_duration_sec for scene in scenes), 3)
        if scene_total_duration != target_duration:
            rules_applied.append(
                "Total planned duration was recomputed from per-scene quantized durations for multi-segment consistency."
            )
            target_duration = scene_total_duration
        takes_per_scene = max((len(scene.takes) for scene in scenes), default=1)
        variations_per_scene = max((len(scene.variations) for scene in scenes), default=1)
        storyboard_candidates_per_scene = max((len(scene.keyframe_candidates) for scene in scenes), default=0)
        storyboard_enabled = any(scene.storyboard_config and scene.storyboard_config.enabled for scene in scenes)
        takes_per_variation = self._determine_take_count(job)
        max_quality_retries_per_scene = self._determine_retry_limit(job)
        storyboard_selection_mode = "preferred_variation_then_first_valid"
        if variations_per_scene > 1:
            rules_applied.append(
                "Each scene now emits controlled shot and prompt variations before take rendering."
            )
        if takes_per_scene > 1:
            rules_applied.append(
                "Each scene renders multiple takes and Phase 2C now prefers technically valid takes before any first-successful tie-break."
            )
        if max_quality_retries_per_scene > 0:
            rules_applied.append(
                "Technically invalid takes can trigger a small capped retry budget per scene."
            )
        if storyboard_enabled:
            rules_applied.append(
                "Optional storyboard keyframes are planned per scene to provide visual pre-steering before video rendering."
            )

        if job.use_storyboard and not storyboard_enabled:
            warnings.append("Storyboard requested but skipped because no storyboard backend is active.")
        if job.use_music:
            warnings.append("Music requested but skipped in Phase 1 because no music backend is active.")

        prompt_text = self._build_prompt(job)
        frame_rate = self.DEFAULT_FRAME_RATE
        steps = [
            ProductionStep(
                name="voice",
                kind="voice",
                adapter_name=voice_capability.name if voice_capability else None,
                enabled=job.use_voice,
                params={
                    "speaker": job.voice_id or None,
                    "language": job.metadata.get("language", "German"),
                    "narration_text_source": "script" if job.script else "idea",
                },
                notes=["Narration uses the script when available, otherwise the idea field."],
                skip_reason=None if job.use_voice else "Voice generation disabled in job input.",
            ),
            ProductionStep(
                name="storyboard",
                kind="storyboard",
                adapter_name=storyboard_capability.name if storyboard_capability else None,
                enabled=storyboard_enabled,
                params={
                    "scene_count": len(scenes),
                    "storyboard_enabled": storyboard_enabled,
                    "candidate_count_per_scene": storyboard_candidates_per_scene,
                    "selection_mode": storyboard_selection_mode,
                    "width": width,
                    "height": height,
                },
                notes=["Storyboard keyframes are optional pre-visualization artifacts and do not replace the current video path."],
                skip_reason=None if storyboard_enabled else "Storyboard disabled or no storyboard backend available.",
            ),
            ProductionStep(
                name="music",
                kind="music",
                adapter_name=music_capability.name if music_capability else None,
                enabled=False,
                skip_reason="Music pipeline reserved for future phases.",
            ),
            ProductionStep(
                name="video",
                kind="video",
                adapter_name=video_capability.name,
                enabled=True,
                input_refs=["voice_audio"] if job.use_voice else [],
                params={
                    "pipeline": selected_pipeline,
                    "render_profile": render_profile,
                    "frame_rate": frame_rate,
                    "num_frames": scenes[0].num_frames if len(scenes) == 1 else None,
                    "planned_duration_sec": target_duration,
                    "scene_count": len(scenes),
                    "segmentation_mode": "multi_scene" if len(scenes) > 1 else "single_scene",
                    "variations_per_scene": variations_per_scene,
                    "storyboard_candidates_per_scene": storyboard_candidates_per_scene,
                    "takes_per_variation": takes_per_variation,
                    "takes_per_scene": takes_per_scene,
                    "selection_mode": "quality_guarded_best_valid_take",
                    "creative_selection_mode": "rule_based_scene_variation_heuristic",
                    "fallback_selection_mode": "first_successful_take",
                    "quality_guard_enabled": True,
                    "creative_selection_enabled": True,
                    "max_quality_retries_per_scene": max_quality_retries_per_scene,
                    "width": width,
                    "height": height,
                    "orientation": job.orientation,
                    "resolution_label": resolution_label,
                },
                notes=["The planned video duration is kept at or above the voice duration to avoid obvious mismatch."],
            ),
        ]

        return ProductionPlan(
            job_id=job.job_id or "",
            orientation=job.orientation or "landscape",
            resolution_label=resolution_label,
            width=width,
            height=height,
            render_profile=render_profile,
            selected_pipeline=selected_pipeline,
            requested_duration_sec=requested_duration,
            target_duration_sec=target_duration,
            estimated_voice_duration_sec=None if actual_voice_duration_sec is not None else estimated_voice_duration,
            actual_voice_duration_sec=round(actual_voice_duration_sec, 3) if actual_voice_duration_sec is not None else None,
            prompt_text=prompt_text,
            warnings=warnings,
            rules_applied=rules_applied,
            scenes=scenes,
            steps=steps,
            metadata={
                "frame_rate": frame_rate,
                "planned_num_frames": snapped_num_frames if len(scenes) == 1 else sum(scene.num_frames for scene in scenes),
                "quantized_target_duration_sec": target_duration,
                "scene_count": len(scenes),
                "segmentation_mode": "multi_scene" if len(scenes) > 1 else "single_scene",
                "variations_per_scene": variations_per_scene,
                "storyboard_enabled": storyboard_enabled,
                "storyboard_candidates_per_scene": storyboard_candidates_per_scene,
                "storyboard_selection_mode": storyboard_selection_mode,
                "storyboard_backend": storyboard_capability.name if storyboard_enabled and storyboard_capability else None,
                "takes_per_variation": takes_per_variation,
                "takes_per_scene": takes_per_scene,
                "selection_mode": "quality_guarded_best_valid_take",
                "creative_selection_mode": "rule_based_scene_variation_heuristic",
                "fallback_selection_mode": "first_successful_take",
                "quality_guard_enabled": True,
                "creative_selection_enabled": True,
                "max_quality_retries_per_scene": max_quality_retries_per_scene,
                "voice_padding_sec": self.VOICE_PADDING_SEC,
                "voice_enabled": job.use_voice,
                "music_requested": job.use_music,
                "storyboard_requested": job.use_storyboard,
            },
        )

    def _resolve_render_profile(self, pipeline_preference: str) -> str:
        if pipeline_preference in {"fast", "balanced", "quality"}:
            return pipeline_preference
        return "balanced"

    def _resolve_pipeline(self, job: JobInput, voice_enabled: bool) -> str:
        requested = job.pipeline_preference
        if requested == "a2vid":
            raise ValueError(
                "pipeline_preference=a2vid is not contract-stable in Phase 1 for generated TTS audio; use auto or ti2vid"
            )
        if requested == "ti2vid":
            return "ti2vid"
        if voice_enabled:
            return "ti2vid"
        return "ti2vid"

    def build_take_render_plan(self, plan: ProductionPlan, scene: ScenePlan, take: TakePlan) -> ProductionPlan:
        video_step = next(step for step in plan.steps if step.name == "video")
        selected_variation = self._variation_for_take(scene, take)
        selected_keyframe = scene.selected_keyframe.model_dump(mode="json") if scene.selected_keyframe else None
        scene_video_step = ProductionStep(
            name="video",
            kind="video",
            adapter_name=video_step.adapter_name,
            enabled=True,
            input_refs=[],
            params={
                "pipeline": plan.selected_pipeline,
                "render_profile": plan.render_profile,
                "frame_rate": take.render_params.get("frame_rate", scene.render_params.get("frame_rate", self.DEFAULT_FRAME_RATE)),
                "num_frames": int(take.render_params.get("num_frames", scene.num_frames)),
                "planned_duration_sec": float(take.render_params.get("planned_duration_sec", scene.target_duration_sec)),
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "scene_title": scene.title,
                "take_id": take.take_id,
                "take_index": take.take_index,
                "variation_id": take.variation_id,
                "variation_index": take.variation_index,
                "shot_type": take.shot_type,
                "camera_style": take.camera_style,
                "camera_motion": take.camera_motion,
                "framing_hint": take.framing_hint,
                "selected_keyframe_path": scene.selected_keyframe.output_path if scene.selected_keyframe else None,
                "selected_keyframe_candidate_id": scene.selected_keyframe.candidate_id if scene.selected_keyframe else None,
                "selection_mode": "quality_guarded_best_valid_take",
                "creative_selection_mode": "rule_based_scene_variation_heuristic",
                "seed": take.seed,
                "width": plan.width,
                "height": plan.height,
                "orientation": plan.orientation,
                "resolution_label": plan.resolution_label,
            },
            notes=[f"Take-specific render plan for {scene.scene_id}/{take.take_id}."],
        )
        return ProductionPlan(
            job_id=plan.job_id,
            orientation=plan.orientation,
            resolution_label=plan.resolution_label,
            width=plan.width,
            height=plan.height,
            render_profile=plan.render_profile,
            selected_pipeline=plan.selected_pipeline,
            requested_duration_sec=float(take.render_params.get("planned_duration_sec", scene.target_duration_sec)),
            target_duration_sec=float(take.render_params.get("planned_duration_sec", scene.target_duration_sec)),
            estimated_voice_duration_sec=None,
            actual_voice_duration_sec=None,
            prompt_text=take.prompt_text,
            warnings=list(plan.warnings),
            rules_applied=list(plan.rules_applied),
            scenes=[
                scene.model_copy(
                    update={
                        "prompt_text": take.prompt_text,
                        "variations": [selected_variation] if selected_variation else list(scene.variations),
                        "selected_keyframe": scene.selected_keyframe,
                        "takes": [take],
                    }
                )
            ],
            steps=[scene_video_step],
            metadata={
                "frame_rate": take.render_params.get("frame_rate", scene.render_params.get("frame_rate", self.DEFAULT_FRAME_RATE)),
                "planned_num_frames": int(take.render_params.get("num_frames", scene.num_frames)),
                "quantized_target_duration_sec": float(take.render_params.get("planned_duration_sec", scene.target_duration_sec)),
                "scene_count": 1,
                "segmentation_mode": "single_scene",
                "source_scene_id": scene.scene_id,
                "source_scene_index": scene.index,
                "variations_per_scene": len(scene.variations) or 1,
                "takes_per_variation": 1,
                "takes_per_scene": 1,
                "take_id": take.take_id,
                "take_index": take.take_index,
                "variation_id": take.variation_id,
                "variation_index": take.variation_index,
                "selected_keyframe": selected_keyframe,
                "selection_mode": "quality_guarded_best_valid_take",
                "creative_selection_mode": "rule_based_scene_variation_heuristic",
                "quality_guard_enabled": True,
                "creative_selection_enabled": True,
            },
        )

    def build_storyboard_render_plan(
        self,
        plan: ProductionPlan,
        scene: ScenePlan,
        candidate: KeyframeCandidatePlan,
    ) -> ProductionPlan:
        storyboard_step = next(step for step in plan.steps if step.name == "storyboard")
        selected_variation = self._variation_for_candidate(scene, candidate)
        scene_storyboard_step = ProductionStep(
            name="storyboard",
            kind="storyboard",
            adapter_name=storyboard_step.adapter_name,
            enabled=True,
            input_refs=[],
            params={
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "scene_title": scene.title,
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "variation_id": candidate.variation_id,
                "variation_index": candidate.variation_index,
                "shot_type": candidate.shot_type,
                "width": candidate.width,
                "height": candidate.height,
                "priority_rank": candidate.priority_rank,
                "relation_type": candidate.relation_type,
                "seed": candidate.render_params.get("seed"),
                "steps": candidate.render_params.get("steps", 9),
                "guidance_scale": candidate.render_params.get("guidance_scale", 0.0),
                "selection_mode": "preferred_variation_then_first_valid",
            },
            notes=[f"Storyboard-specific render plan for {scene.scene_id}/{candidate.candidate_id}."],
        )
        return ProductionPlan(
            job_id=plan.job_id,
            orientation=plan.orientation,
            resolution_label=plan.resolution_label,
            width=plan.width,
            height=plan.height,
            render_profile=plan.render_profile,
            selected_pipeline=plan.selected_pipeline,
            requested_duration_sec=scene.target_duration_sec,
            target_duration_sec=scene.target_duration_sec,
            estimated_voice_duration_sec=None,
            actual_voice_duration_sec=None,
            prompt_text=candidate.prompt_text,
            warnings=list(plan.warnings),
            rules_applied=list(plan.rules_applied),
            scenes=[
                scene.model_copy(
                    update={
                        "prompt_text": candidate.prompt_text,
                        "variations": [selected_variation] if selected_variation else list(scene.variations),
                        "keyframe_candidates": [candidate],
                        "selected_keyframe": None,
                        "takes": [],
                    }
                )
            ],
            steps=[scene_storyboard_step],
            metadata={
                "scene_count": 1,
                "segmentation_mode": "single_scene",
                "source_scene_id": scene.scene_id,
                "source_scene_index": scene.index,
                "storyboard_enabled": True,
                "storyboard_selection_mode": "preferred_variation_then_first_valid",
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "variation_id": candidate.variation_id,
                "variation_index": candidate.variation_index,
                "width": candidate.width,
                "height": candidate.height,
            },
        )

    def _build_prompt(self, job: JobInput) -> str:
        parts = [job.script or job.idea]
        if job.style:
            parts.append(f"Style direction: {job.style}.")
        if job.extra_llm_instruction:
            parts.append(f"Extra instruction: {job.extra_llm_instruction}.")
        return " ".join(part.strip() for part in parts if part.strip())

    def _build_scene_plans(
        self,
        job: JobInput,
        *,
        width: int,
        height: int,
        resolution_label: str,
        render_profile: str,
        selected_pipeline: str,
        target_duration_sec: float,
        frame_rate: int,
    ) -> list[ScenePlan]:
        text_units = self._split_text_units(job.primary_text)
        scene_count = self._determine_scene_count(job, target_duration_sec, len(text_units))
        grouped_units = self._group_text_units(text_units, scene_count, fallback_text=job.idea or job.script or "visual beat")
        raw_scene_durations = self._allocate_scene_durations(grouped_units, target_duration_sec)
        variations_per_scene = self._determine_variation_count(job)
        takes_per_variation = self._determine_take_count(job)
        storyboard_candidate_count = self._determine_storyboard_candidate_count(job)
        storyboard_enabled = bool(job.use_storyboard and self.registry.primary_capability("storyboard"))

        scenes: list[ScenePlan] = []
        narration_cursor = 0.0
        for index, (scene_text, raw_duration_sec) in enumerate(zip(grouped_units, raw_scene_durations), start=1):
            num_frames, quantized_duration_sec = quantize_duration_to_frame_contract(raw_duration_sec, frame_rate)
            scene_id = f"scene_{index:02d}"
            title = f"Scene {index}"
            description = self._scene_description(scene_text, job.idea or job.script or "visual beat")
            prompt_text = self._scene_prompt(job, description, scene_text, index, scene_count)
            narration_text = scene_text if job.use_voice else None
            narration_start_sec = round(narration_cursor, 3) if narration_text else None
            narration_end_sec = round(narration_cursor + quantized_duration_sec, 3) if narration_text else None
            narration_cursor = round(narration_cursor + quantized_duration_sec, 3)
            render_params = {
                "pipeline": selected_pipeline,
                "render_profile": render_profile,
                "frame_rate": frame_rate,
                "num_frames": num_frames,
                "planned_duration_sec": quantized_duration_sec,
                "width": width,
                "height": height,
                "orientation": job.orientation or "landscape",
                "resolution_label": resolution_label,
            }
            shot = ShotPlan(
                shot_id=f"{scene_id}_shot_01",
                scene_id=scene_id,
                index=1,
                description=description,
                target_duration_sec=quantized_duration_sec,
                num_frames=num_frames,
                prompt_text=prompt_text,
                narration_text=narration_text,
                narration_start_sec=narration_start_sec,
                narration_end_sec=narration_end_sec,
                render_params=render_params,
                notes=["Phase 2A uses one renderable shot per scene as the minimal structured planning contract."],
            )
            variation_plans = self._build_variation_plans(
                job,
                scene_id=scene_id,
                scene_index=index,
                scene_count=scene_count,
                description=description,
                scene_prompt_text=prompt_text,
                variation_count=variations_per_scene,
            )
            storyboard_config = self._build_storyboard_config(
                scene_id=scene_id,
                scene_index=index,
                scene_count=scene_count,
                scene_text=scene_text,
                variations=variation_plans,
                enabled=storyboard_enabled,
                requested_candidate_count=storyboard_candidate_count,
            )
            keyframe_candidates = self._build_keyframe_candidate_plans(
                job,
                scene_id=scene_id,
                scene_index=index,
                scene_count=scene_count,
                variations=variation_plans,
                scene_prompt_text=prompt_text,
                width=width,
                height=height,
                storyboard_config=storyboard_config,
            )
            take_plans = self._build_take_plans(
                job,
                scene_id=scene_id,
                scene_index=index,
                variations=variation_plans,
                takes_per_variation=takes_per_variation,
                render_params=render_params,
            )
            scenes.append(
                ScenePlan(
                    scene_id=scene_id,
                    index=index,
                    title=title,
                    description=description,
                    target_duration_sec=quantized_duration_sec,
                    num_frames=num_frames,
                    prompt_text=prompt_text,
                    narration_text=narration_text,
                    narration_start_sec=narration_start_sec,
                    narration_end_sec=narration_end_sec,
                    render_params=render_params,
                    shots=[shot],
                    variations=variation_plans,
                    storyboard_config=storyboard_config,
                    keyframe_candidates=keyframe_candidates,
                    selected_keyframe=None,
                    takes=take_plans,
                    notes=[
                        "Scene duration is quantized independently to the LTX frame contract.",
                        f"Phase 2D builds {len(variation_plans)} controlled creative variation(s) for this scene.",
                        (
                            f"Phase 3A plans {len(keyframe_candidates)} optional storyboard keyframe candidate(s) for this scene."
                            if storyboard_config.enabled
                            else "Phase 3A storyboard is inactive for this scene."
                        ),
                    ],
                )
            )
        return scenes

    def _determine_scene_count(self, job: JobInput, target_duration_sec: float, unit_count: int) -> int:
        metadata = job.metadata or {}
        if metadata.get("force_single_scene"):
            return 1

        explicit_scene_count = metadata.get("scene_count")
        if explicit_scene_count is not None:
            try:
                count = int(explicit_scene_count)
            except (TypeError, ValueError):
                count = 1
            max_count = max(1, min(self.MAX_SCENES, int(target_duration_sec // self.MIN_SCENE_DURATION_SEC) or 1))
            return max(1, min(count, max_count))

        max_count = max(1, min(self.MAX_SCENES, int(target_duration_sec // self.MIN_SCENE_DURATION_SEC) or 1))
        if target_duration_sec <= 5.5:
            auto_count = 1
        elif target_duration_sec <= 9.0:
            auto_count = 2
        elif target_duration_sec <= 13.5:
            auto_count = 3
        elif target_duration_sec <= 18.0:
            auto_count = 4
        else:
            auto_count = 5

        if unit_count <= 1 and target_duration_sec <= 6.5:
            auto_count = 1

        return max(1, min(auto_count, max_count))

    def _split_text_units(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _determine_take_count(self, job: JobInput) -> int:
        requested = job.metadata.get("takes_per_scene", 1)
        try:
            take_count = int(requested)
        except (TypeError, ValueError):
            take_count = 1
        return max(1, min(take_count, self.MAX_TAKES_PER_SCENE))

    def _determine_variation_count(self, job: JobInput) -> int:
        requested = job.metadata.get("variations_per_scene", 1)
        try:
            variation_count = int(requested)
        except (TypeError, ValueError):
            variation_count = 1
        return max(1, min(variation_count, self.MAX_VARIATIONS_PER_SCENE))

    def _determine_retry_limit(self, job: JobInput) -> int:
        requested = job.metadata.get("max_take_retries_per_scene", 1)
        try:
            retry_limit = int(requested)
        except (TypeError, ValueError):
            retry_limit = 1
        return max(0, min(retry_limit, self.MAX_TAKE_RETRIES_PER_SCENE))

    def _determine_storyboard_candidate_count(self, job: JobInput) -> int:
        default_count = 2 if job.use_storyboard else 0
        requested = job.metadata.get("storyboard_candidates_per_scene", default_count)
        try:
            candidate_count = int(requested)
        except (TypeError, ValueError):
            candidate_count = default_count
        return max(0, min(candidate_count, self.MAX_STORYBOARD_CANDIDATES))

    def _group_text_units(self, units: list[str], scene_count: int, *, fallback_text: str) -> list[str]:
        if scene_count <= 1:
            return [" ".join(units).strip() or fallback_text]

        if not units:
            return [fallback_text for _ in range(scene_count)]

        groups = ["" for _ in range(scene_count)]
        for index, unit in enumerate(units):
            group_index = min(scene_count - 1, math.floor(index * scene_count / max(len(units), 1)))
            groups[group_index] = " ".join(part for part in [groups[group_index], unit] if part).strip()

        fallback = fallback_text or units[-1]
        return [group or fallback for group in groups]

    def _allocate_scene_durations(self, grouped_units: list[str], total_duration_sec: float) -> list[float]:
        scene_count = len(grouped_units)
        if scene_count == 1:
            return [total_duration_sec]

        minimum_total = scene_count * self.MIN_SCENE_DURATION_SEC
        base_duration = max(total_duration_sec, minimum_total)
        residual = max(0.0, base_duration - minimum_total)
        weights = [max(1.0, len(group.split())) for group in grouped_units]
        weight_total = sum(weights) or float(scene_count)

        durations = []
        for weight in weights:
            durations.append(self.MIN_SCENE_DURATION_SEC + residual * (weight / weight_total))
        return durations

    def _scene_description(self, scene_text: str, fallback_text: str) -> str:
        source = scene_text or fallback_text
        normalized = " ".join(source.split()).strip()
        return normalized[:180] or "Visual beat"

    def _scene_prompt(self, job: JobInput, description: str, scene_text: str, index: int, scene_count: int) -> str:
        prompt_parts = [
            description,
            f"Scene {index} of {scene_count}.",
        ]
        if job.style:
            prompt_parts.append(f"Style direction: {job.style}.")
        if scene_text:
            prompt_parts.append(f"Narrative focus: {scene_text}.")
        if job.extra_llm_instruction:
            prompt_parts.append(f"Extra instruction: {job.extra_llm_instruction}.")
        return " ".join(part.strip() for part in prompt_parts if part.strip())

    def _build_variation_plans(
        self,
        job: JobInput,
        *,
        scene_id: str,
        scene_index: int,
        scene_count: int,
        description: str,
        scene_prompt_text: str,
        variation_count: int,
    ) -> list[VariationPlan]:
        presets = self._variation_blueprints(scene_index=scene_index, scene_count=scene_count)
        selected_presets = presets[:variation_count]
        variations: list[VariationPlan] = []
        for variation_index, preset in enumerate(selected_presets, start=1):
            variation_id = f"{scene_id}_var_{variation_index:02d}"
            prompt_delta = str(preset["prompt_delta"])
            prompt_variant_text = self._compose_variation_prompt(
                scene_prompt_text=scene_prompt_text,
                shot_type=str(preset["shot_type"]),
                camera_style=preset.get("camera_style"),
                camera_motion=preset.get("camera_motion"),
                framing_hint=str(preset["framing_hint"]),
                prompt_delta=prompt_delta,
                style_bias=preset.get("style_bias"),
            )
            variations.append(
                VariationPlan(
                    variation_id=variation_id,
                    scene_id=scene_id,
                    variation_index=variation_index,
                    shot_type=str(preset["shot_type"]),
                    camera_style=str(preset["camera_style"]) if preset.get("camera_style") else None,
                    camera_motion=str(preset["camera_motion"]) if preset.get("camera_motion") else None,
                    framing_hint=str(preset["framing_hint"]),
                    prompt_delta=prompt_delta,
                    prompt_variant_text=prompt_variant_text,
                    style_bias=str(preset["style_bias"]) if preset.get("style_bias") else None,
                    notes=[
                        f"Phase 2D variation {variation_index} for {scene_id}.",
                        f"Base scene description: {description}",
                    ],
                )
            )
        return variations

    def _variation_blueprints(self, *, scene_index: int, scene_count: int) -> list[dict[str, str]]:
        return [
            {
                "shot_type": "establishing",
                "camera_motion": "slow push-in",
                "framing_hint": "wide environmental framing with a readable subject anchor",
                "prompt_delta": "Emphasize geography, context and a strong opening silhouette with a controlled slow push-in.",
                "style_bias": "scale",
            },
            {
                "shot_type": "medium_action",
                "camera_motion": "gentle lateral tracking",
                "framing_hint": "medium three-quarter framing with clearer subject focus",
                "prompt_delta": "Bring the scene closer to the subject with a medium composition and subtle lateral movement.",
                "style_bias": "motion",
            },
            {
                "shot_type": "detail_closeup",
                "camera_style": "intimate cinematic close-up",
                "framing_hint": "tight detail framing with tactile surface emphasis",
                "prompt_delta": "Prioritize material detail, interfaces, lighting accents and a tighter crop without losing scene coherence.",
                "style_bias": "texture",
            },
            {
                "shot_type": "hero_tableau",
                "camera_style": "composed hero frame",
                "framing_hint": "balanced hero framing with deliberate negative space",
                "prompt_delta": (
                    f"Present scene {scene_index} of {scene_count} as a clearer hero image with calmer motion and stronger readability."
                ),
                "style_bias": "clarity",
            },
        ]

    def _compose_variation_prompt(
        self,
        *,
        scene_prompt_text: str,
        shot_type: str,
        camera_style: Any,
        camera_motion: Any,
        framing_hint: str,
        prompt_delta: str,
        style_bias: Any,
    ) -> str:
        prompt_parts = [scene_prompt_text]
        prompt_parts.append(f"Shot variation: {shot_type}.")
        if camera_style:
            prompt_parts.append(f"Camera style: {camera_style}.")
        if camera_motion:
            prompt_parts.append(f"Camera motion: {camera_motion}.")
        prompt_parts.append(f"Framing hint: {framing_hint}.")
        prompt_parts.append(f"Prompt delta: {prompt_delta}.")
        if style_bias:
            prompt_parts.append(f"Style bias: {style_bias}.")
        return " ".join(part.strip() for part in prompt_parts if part and str(part).strip())

    def _build_storyboard_config(
        self,
        *,
        scene_id: str,
        scene_index: int,
        scene_count: int,
        scene_text: str,
        variations: list[VariationPlan],
        enabled: bool,
        requested_candidate_count: int,
    ) -> StoryboardConfig:
        if not enabled or requested_candidate_count <= 0:
            return StoryboardConfig(scene_id=scene_id, enabled=False, candidate_count=0)

        ranked_variations = self._rank_storyboard_variations(
            variations=variations,
            scene_index=scene_index,
            scene_count=scene_count,
            scene_text=scene_text,
        )
        preferred = ranked_variations[0] if ranked_variations else None
        return StoryboardConfig(
            scene_id=scene_id,
            enabled=True,
            required=False,
            candidate_count=min(requested_candidate_count, len(ranked_variations) or 1),
            preferred_variation_id=preferred.variation_id if preferred else None,
            preferred_variation_index=preferred.variation_index if preferred else None,
            priority_rule=self._storyboard_priority_rule(
                preferred.shot_type if preferred else None,
                scene_index,
                scene_count,
                scene_text,
            ),
            selection_mode="preferred_variation_then_first_valid",
            notes=["Phase 3A uses small rule-based storyboard planning per scene."],
        )

    def _build_keyframe_candidate_plans(
        self,
        job: JobInput,
        *,
        scene_id: str,
        scene_index: int,
        scene_count: int,
        variations: list[VariationPlan],
        scene_prompt_text: str,
        width: int,
        height: int,
        storyboard_config: StoryboardConfig,
    ) -> list[KeyframeCandidatePlan]:
        if not storyboard_config.enabled or storyboard_config.candidate_count <= 0:
            return []

        ranked_variations = self._rank_storyboard_variations(
            variations=variations,
            scene_index=scene_index,
            scene_count=scene_count,
            scene_text=scene_prompt_text,
        )
        selected_variations = ranked_variations[: storyboard_config.candidate_count]
        zimage_overrides = job.backend_overrides.get("zimage", {})
        candidates: list[KeyframeCandidatePlan] = []
        for candidate_index, variation in enumerate(selected_variations, start=1):
            candidate_id = f"{variation.variation_id}_keyframe_{candidate_index:02d}"
            seed = stable_seed(f"{job.job_id or job.primary_text}:{candidate_id}:storyboard")
            prompt_text = (
                f"{variation.prompt_variant_text} Storyboard keyframe still image. "
                "One clean representative frame, sharp composition, no motion blur, no text overlay."
            )
            candidates.append(
                KeyframeCandidatePlan(
                    candidate_id=candidate_id,
                    scene_id=scene_id,
                    candidate_index=candidate_index,
                    variation_id=variation.variation_id,
                    variation_index=variation.variation_index,
                    shot_type=variation.shot_type,
                    prompt_text=prompt_text,
                    width=width,
                    height=height,
                    priority_rank=candidate_index,
                    relation_type="scene_variation",
                    render_params={
                        "seed": seed,
                        "steps": int(zimage_overrides.get("steps", 9)),
                        "guidance_scale": float(zimage_overrides.get("guidance_scale", 0.0)),
                    },
                    notes=[
                        f"Phase 3A storyboard candidate {candidate_index} for {scene_id}.",
                        f"Derived from variation {variation.variation_id}.",
                    ],
                )
            )
        return candidates

    def _rank_storyboard_variations(
        self,
        *,
        variations: list[VariationPlan],
        scene_index: int,
        scene_count: int,
        scene_text: str,
    ) -> list[VariationPlan]:
        def sort_key(variation: VariationPlan) -> tuple[int, int]:
            return (
                -self._storyboard_variation_score(variation, scene_index, scene_count, scene_text),
                variation.variation_index,
            )

        return sorted(variations, key=sort_key)

    def _storyboard_variation_score(
        self,
        variation: VariationPlan,
        scene_index: int,
        scene_count: int,
        scene_text: str,
    ) -> int:
        score = 0
        text = scene_text.lower()
        shot_type = variation.shot_type
        if scene_count <= 1 or scene_index == 1:
            if shot_type == "establishing":
                score += 6
            elif shot_type == "hero_tableau":
                score += 4
        elif scene_index == scene_count:
            if shot_type == "hero_tableau":
                score += 6
            elif shot_type == "establishing":
                score += 3
        else:
            if shot_type == "medium_action":
                score += 5
            elif shot_type == "detail_closeup":
                score += 3

        if any(token in text for token in {"render", "progress", "moving", "motion"}):
            if shot_type == "medium_action":
                score += 3
        if any(token in text for token in {"boot", "wake", "open", "startup"}):
            if shot_type == "establishing":
                score += 2
        if any(token in text for token in {"detail", "interface", "panel", "surface"}):
            if shot_type == "detail_closeup":
                score += 2
        if any(token in text for token in {"final", "clean", "resolve", "complete"}):
            if shot_type in {"hero_tableau", "establishing"}:
                score += 2

        return score

    def _storyboard_priority_rule(
        self,
        shot_type: str | None,
        scene_index: int,
        scene_count: int,
        scene_text: str,
    ) -> str:
        if shot_type == "establishing" and (scene_count <= 1 or scene_index == 1):
            return "opening_prefers_establishing_keyframe"
        if shot_type == "hero_tableau" and scene_index == scene_count:
            return "final_prefers_hero_keyframe"
        if shot_type == "medium_action" and any(token in scene_text.lower() for token in {"render", "progress", "moving", "motion"}):
            return "motion_scene_prefers_medium_keyframe"
        if shot_type == "detail_closeup":
            return "detail_scene_prefers_closeup_keyframe"
        return "variation_priority_rank"

    def _build_take_plans(
        self,
        job: JobInput,
        *,
        scene_id: str,
        scene_index: int,
        variations: list[VariationPlan],
        takes_per_variation: int,
        render_params: dict[str, object],
    ) -> list[TakePlan]:
        takes: list[TakePlan] = []
        base_seed_material = f"{job.job_id or job.primary_text}:{scene_id}"
        take_index = 0
        use_legacy_take_ids = len(variations) == 1
        for variation in variations:
            for variation_take_index in range(1, takes_per_variation + 1):
                take_index += 1
                take_id = (
                    f"{scene_id}_take_{take_index:02d}"
                    if use_legacy_take_ids
                    else f"{variation.variation_id}_take_{variation_take_index:02d}"
                )
                seed = stable_seed(f"{base_seed_material}:{variation.variation_id}:{variation_take_index}")
                take_render_params = dict(render_params)
                take_render_params.update(
                    {
                        "seed": seed,
                        "take_id": take_id,
                        "take_index": take_index,
                        "variation_id": variation.variation_id,
                        "variation_index": variation.variation_index,
                        "variation_take_index": variation_take_index,
                        "shot_type": variation.shot_type,
                        "camera_style": variation.camera_style,
                        "camera_motion": variation.camera_motion,
                        "framing_hint": variation.framing_hint,
                        "prompt_variant_text": variation.prompt_variant_text,
                        "style_bias": variation.style_bias,
                    }
                )
                takes.append(
                    TakePlan(
                        take_id=take_id,
                        scene_id=scene_id,
                        take_index=take_index,
                        variation_id=variation.variation_id,
                        variation_index=variation.variation_index,
                        shot_type=variation.shot_type,
                        camera_style=variation.camera_style,
                        camera_motion=variation.camera_motion,
                        framing_hint=variation.framing_hint,
                        prompt_variant_text=variation.prompt_variant_text,
                        style_bias=variation.style_bias,
                        seed=seed,
                        prompt_text=variation.prompt_variant_text,
                        render_params=take_render_params,
                        notes=[
                            f"Phase 2D take for variation {variation.variation_id}.",
                            "Phase 2B renders multiple takes per scene.",
                            "Phase 2C validates each take technically before selection.",
                        ],
                    )
                )
        return takes

    @staticmethod
    def _variation_for_take(scene: ScenePlan, take: TakePlan) -> VariationPlan | None:
        if not take.variation_id:
            return None
        for variation in scene.variations:
            if variation.variation_id == take.variation_id:
                return variation
        return None

    @staticmethod
    def _variation_for_candidate(scene: ScenePlan, candidate: KeyframeCandidatePlan) -> VariationPlan | None:
        if not candidate.variation_id:
            return None
        for variation in scene.variations:
            if variation.variation_id == candidate.variation_id:
                return variation
        return None
