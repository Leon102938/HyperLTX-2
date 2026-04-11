from __future__ import annotations

import time
from pathlib import Path

from agent_core.adapters.base import VideoAdapter
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.utils import frame_count_to_duration_sec, http_json, probe_media_duration


class LTX2Adapter(VideoAdapter):
    name = "ltx2"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        poll_interval_sec: float = 5.0,
        max_wait_sec: float = 3600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval_sec = poll_interval_sec
        self.max_wait_sec = max_wait_sec

    def capabilities(self) -> BackendCapabilities:
        notes = ["uses local FastAPI /ltx2 endpoints"]
        try:
            ready_payload = http_json(f"{self.base_url}/DW/ready")
            health_payload = http_json(f"{self.base_url}/health")
            available = bool(ready_payload.get("ready")) and bool(health_payload.get("status") == "ok")
        except Exception as exc:
            available = False
            notes.append(f"readiness probe failed: {exc}")

        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=available,
            phase1_enabled=True,
            transport="http",
            supported_pipelines=["ti2vid"],
            supported_orientations=["landscape", "portrait", "square"],
            supported_resolution_labels=["draft", "standard", "high", "custom"],
            notes=notes + ["Underlying backend supports more modes, but Phase 1 core uses ti2vid as the stable contract."],
        )

    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        video_step = next(step for step in plan.steps if step.name == "video")
        frame_rate = int(video_step.params.get("frame_rate") or plan.metadata.get("frame_rate", 24))
        planned_num_frames = int(video_step.params.get("num_frames") or plan.metadata.get("planned_num_frames") or 0)
        if planned_num_frames <= 0:
            raise RuntimeError("Production plan is missing a valid num_frames contract for the video step")

        overrides = {
            "pipeline": plan.selected_pipeline,
            "width": plan.width,
            "height": plan.height,
            "frame_rate": frame_rate,
            "num_frames": planned_num_frames,
            "enhance_prompt": plan.render_profile == "quality",
        }
        if video_step.params.get("seed") is not None:
            overrides["seed"] = int(video_step.params["seed"])

        if plan.render_profile == "fast":
            overrides["num_inference_steps"] = 8
        elif plan.render_profile == "quality":
            overrides["num_inference_steps"] = 18
        else:
            overrides["num_inference_steps"] = 12

        if plan.selected_pipeline == "a2vid" and voice_result and voice_result.output_path:
            overrides["audio_path"] = voice_result.output_path

        overrides.update(job.backend_overrides.get("ltx2", {}))
        effective_num_frames = int(overrides.get("num_frames", planned_num_frames))
        expected_duration_sec = frame_count_to_duration_sec(effective_num_frames, frame_rate)
        payload = {
            "job_id": f"{job.job_id}_video",
            "prompt": plan.prompt_text,
            "overrides": overrides,
        }

        submit = http_json(f"{self.base_url}/ltx2/submit", method="POST", payload=payload, timeout=60)
        backend_job_id = str(submit["job_id"])
        status_url = f"{self.base_url}/ltx2/status/{backend_job_id}"
        get_url = f"{self.base_url}/ltx2/get/{backend_job_id}"

        deadline = time.monotonic() + self.max_wait_sec
        latest_status: dict[str, object] = {}
        while time.monotonic() < deadline:
            latest_status = http_json(status_url, timeout=60)
            status = str(latest_status.get("status") or latest_status.get("state") or "")
            if latest_status.get("error") == "not found":
                raise RuntimeError(f"LTX2 job {backend_job_id} disappeared before completion")
            if status in {"succeeded", "failed"}:
                break
            time.sleep(self.poll_interval_sec)
        else:
            raise RuntimeError(f"LTX2 job {backend_job_id} timed out after {self.max_wait_sec}s")

        result_payload = http_json(get_url, timeout=60)
        if not result_payload.get("ok"):
            error = str(result_payload.get("error") or latest_status.get("error") or "ltx2 failed")
            return ExecutionResult(
                step_name="video",
                success=False,
                status="failed",
                backend_name=self.name,
                backend_job_id=backend_job_id,
                error=error,
                metadata={"submit": submit, "status": latest_status, "result": result_payload},
            )

        output_path = str(result_payload.get("output_path") or "")
        duration_sec = probe_media_duration(output_path)
        artifacts = []
        if output_path:
            path_obj = Path(output_path)
            artifacts.append(
                ArtifactRef(
                    key="final_video",
                    kind="video",
                    path=str(path_obj),
                    origin=self.name,
                    exists=path_obj.exists(),
                    metadata={
                        "video_url": result_payload.get("video_url"),
                        "backend": result_payload.get("backend"),
                        "planned_duration_sec": plan.target_duration_sec,
                        "expected_duration_sec": expected_duration_sec,
                        "planned_num_frames": planned_num_frames,
                        "effective_num_frames": effective_num_frames,
                    },
                )
            )

        return ExecutionResult(
            step_name="video",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=backend_job_id,
            output_path=output_path or None,
            output_url=result_payload.get("video_url"),
            duration_sec=duration_sec,
            artifacts=artifacts,
            metadata={
                "submit": submit,
                "status": latest_status,
                "result": result_payload,
                "duration_contract": {
                    "planned_duration_sec": plan.target_duration_sec,
                    "expected_duration_sec": expected_duration_sec,
                    "planned_num_frames": planned_num_frames,
                    "effective_num_frames": effective_num_frames,
                    "frame_rate": frame_rate,
                },
            },
        )
