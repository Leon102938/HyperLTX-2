from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from agent_core.adapters.base import VideoAdapter, VoiceAdapter
from agent_core.assembler import ResultAssembler
from agent_core.backend_registry import BackendRegistry, build_default_registry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import ArtifactRef, ExecutionResult, JobInput, ProductionPlan, ResultSummary, TakeResultRecord
from agent_core.state_store import StateStore
from agent_core.utils import build_job_id, mirror_media_file, read_json


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
            )
            self.state_store.transition(state, "assembled", "Result assembled.")
            self.state_store.save_result(state, result)
            self.state_store.transition(state, "done", "Job finished successfully.")
            self.state_store.save_result(state, result)
            return result
        except Exception as exc:
            self.state_store.append_log(state.job_id, traceback.format_exc())
            self.state_store.fail(state, str(exc))
            failed_result = self.assembler.failure(job, plan, state, str(exc), voice_result=voice_result)
            self.state_store.save_result(state, failed_result)
            if raise_on_error:
                raise
            return failed_result

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
        if len(plan.scenes) <= 1 and planned_takes_per_scene <= 1:
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

        for scene in plan.scenes:
            self.state_store.append_log(state.job_id, f"scene {scene.scene_id} started")
            scene_workspace = workspace / "scenes" / scene.scene_id
            take_records: list[TakeResultRecord] = []
            selected_take: TakeResultRecord | None = None
            for take in scene.takes:
                self.state_store.append_log(state.job_id, f"take {take.take_id} started")
                take_job = job.model_copy(
                    update={
                        "job_id": f"{job.job_id}_{take.take_id}",
                        "metadata": {
                            **job.metadata,
                            "scene_id": scene.scene_id,
                            "scene_index": scene.index,
                            "take_id": take.take_id,
                            "take_index": take.take_index,
                            "take_seed": take.seed,
                        },
                    }
                )
                take_workspace = scene_workspace / "takes" / take.take_id
                take_plan = self.planner.build_take_render_plan(plan, scene, take)
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
                                "seed": take.seed,
                                "source_output_path": take_result.output_path,
                            },
                        )
                    )
                take_record = TakeResultRecord(
                    take_id=take.take_id,
                    scene_id=scene.scene_id,
                    take_index=take.take_index,
                    seed=take.seed,
                    status=take_result.status,
                    output_path=mirrored_output_path or take_result.output_path,
                    output_url=take_result.output_url,
                    duration_sec=take_result.duration_sec,
                    selected=False,
                    metadata={
                        "prompt_text": take.prompt_text,
                        "backend_metadata": take_result.metadata,
                    },
                    error=take_result.error,
                )
                if take_result.success and take_record.output_path and selected_take is None:
                    take_record.selected = True
                    selected_take = take_record
                take_records.append(take_record)
                self.state_store.append_log(state.job_id, f"take {take.take_id} finished with status={take_result.status}")

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
                        "takes_per_scene": planned_takes_per_scene,
                        "selection_mode": "first_successful_take",
                        "scene_outputs": scene_outputs
                        + [
                            {
                                "scene_id": scene.scene_id,
                                "scene_index": scene.index,
                                "title": scene.title,
                                "prompt_text": scene.prompt_text,
                                "selected_take_id": None,
                                "selected_take": None,
                                "takes": [record.model_dump(mode="json") for record in take_records],
                            }
                        ],
                    },
                    error=f"all takes failed for {scene.scene_id}",
                )

            aggregate_duration_sec += selected_take.duration_sec or scene.target_duration_sec
            scene_output = {
                "scene_id": scene.scene_id,
                "scene_index": scene.index,
                "title": scene.title,
                "output_path": selected_take.output_path,
                "output_url": selected_take.output_url,
                "duration_sec": selected_take.duration_sec or scene.target_duration_sec,
                "prompt_text": scene.prompt_text,
                "selected_take_id": selected_take.take_id,
                "selected_take": selected_take.model_dump(mode="json"),
                "take_count": len(take_records),
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
                        },
                    )
                )
            self.state_store.append_log(
                state.job_id,
                f"scene {scene.scene_id} finished with selected_take={selected_take.take_id}",
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
                "takes_per_scene": planned_takes_per_scene,
                "selection_mode": "first_successful_take",
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
            "takes_per_scene": video_result.metadata.get("takes_per_scene", 1),
            "selection_mode": video_result.metadata.get("selection_mode", "first_successful_take"),
            "scene_outputs": video_result.metadata.get("scene_outputs", []),
            "selected_scene_outputs": video_result.metadata.get("selected_scene_outputs", []),
        }
