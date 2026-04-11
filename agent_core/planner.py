from __future__ import annotations

import math
import re

from agent_core.backend_registry import BackendRegistry
from agent_core.schemas import JobInput, ProductionPlan, ProductionStep, ScenePlan, ShotPlan, TakePlan
from agent_core.utils import choose_resolution, estimate_speech_duration, quantize_duration_to_frame_contract, stable_seed


class ProductionPlanner:
    MIN_VIDEO_DURATION_SEC = 4.0
    MIN_SCENE_DURATION_SEC = 2.0
    MAX_SCENES = 6
    MAX_TAKES_PER_SCENE = 4
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
        if takes_per_scene > 1:
            rules_applied.append(
                "Each scene renders multiple takes and currently selects the first successful take as the stable default."
            )

        if job.use_storyboard:
            warnings.append("Storyboard requested but skipped in Phase 1 because no storyboard backend is active.")
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
                enabled=False,
                skip_reason="Storyboard reserved for future phases.",
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
                    "takes_per_scene": takes_per_scene,
                    "selection_mode": "first_successful_take",
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
                "takes_per_scene": takes_per_scene,
                "selection_mode": "first_successful_take",
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
                "selection_mode": "first_successful_take",
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
                "takes_per_scene": 1,
                "take_id": take.take_id,
                "take_index": take.take_index,
                "selection_mode": "first_successful_take",
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
        takes_per_scene = self._determine_take_count(job)

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
            take_plans = self._build_take_plans(
                job,
                scene_id=scene_id,
                scene_index=index,
                take_count=takes_per_scene,
                prompt_text=prompt_text,
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
                    takes=take_plans,
                    notes=["Scene duration is quantized independently to the LTX frame contract."],
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

    def _build_take_plans(
        self,
        job: JobInput,
        *,
        scene_id: str,
        scene_index: int,
        take_count: int,
        prompt_text: str,
        render_params: dict[str, object],
    ) -> list[TakePlan]:
        takes: list[TakePlan] = []
        base_seed_material = f"{job.job_id or job.primary_text}:{scene_id}"
        for take_index in range(1, take_count + 1):
            take_id = f"{scene_id}_take_{take_index:02d}"
            seed = stable_seed(f"{base_seed_material}:{take_index}")
            take_render_params = dict(render_params)
            take_render_params["seed"] = seed
            take_render_params["take_id"] = take_id
            take_render_params["take_index"] = take_index
            takes.append(
                TakePlan(
                    take_id=take_id,
                    scene_id=scene_id,
                    take_index=take_index,
                    seed=seed,
                    prompt_text=prompt_text,
                    render_params=take_render_params,
                    notes=[
                        "Phase 2B renders multiple takes per scene.",
                        "Selection currently prefers the first successful take.",
                    ],
                )
            )
        return takes
