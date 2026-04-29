from __future__ import annotations

import time
from pathlib import Path

from agent_core.adapters.base import StoryboardAdapter
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.utils import compress_visual_prompt, http_json


class ZImageStoryboardAdapter(StoryboardAdapter):
    name = "zimage_storyboard"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        poll_interval_sec: float = 2.0,
        max_wait_sec: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval_sec = poll_interval_sec
        self.max_wait_sec = max_wait_sec

    def capabilities(self) -> BackendCapabilities:
        notes = ["uses local FastAPI /zimage endpoints for storyboard keyframes"]
        try:
            payload = http_json(f"{self.base_url}/DW/zimage_ready")
            available = bool(payload.get("ready"))
        except Exception as exc:
            available = False
            notes.append(f"readiness probe failed: {exc}")

        return BackendCapabilities(
            name=self.name,
            kind="storyboard",
            available=available,
            phase1_enabled=True,
            transport="http",
            supported_orientations=["landscape", "portrait", "square"],
            supported_resolution_labels=["draft", "standard", "high", "custom"],
            notes=notes,
        )

    def generate_storyboard(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        storyboard_step = next(step for step in plan.steps if step.name == "storyboard")
        effective_prompt, prompt_source = self._resolve_effective_prompt(plan, storyboard_step)
        payload = {
            "job_id": str(storyboard_step.params.get("candidate_id") or f"{job.job_id}_storyboard"),
            "prompt": effective_prompt,
            "width": int(storyboard_step.params.get("width", plan.width)),
            "height": int(storyboard_step.params.get("height", plan.height)),
            "steps": int(storyboard_step.params.get("steps", 9)),
            "guidance_scale": float(storyboard_step.params.get("guidance_scale", 0.0)),
            "seed": int(storyboard_step.params.get("seed")) if storyboard_step.params.get("seed") is not None else None,
        }
        submit = http_json(f"{self.base_url}/zimage/jobs", method="POST", payload=payload, timeout=60)
        backend_job_id = str(submit["job_id"])
        status_url = f"{self.base_url}/zimage/jobs/{backend_job_id}"

        deadline = time.monotonic() + self.max_wait_sec
        latest_status: dict[str, object] = {}
        while time.monotonic() < deadline:
            latest_status = http_json(status_url, timeout=60)
            status = str(latest_status.get("state") or "")
            if status in {"succeeded", "failed"}:
                break
            time.sleep(self.poll_interval_sec)
        else:
            raise RuntimeError(f"Z-Image storyboard job {backend_job_id} timed out after {self.max_wait_sec}s")

        if str(latest_status.get("state")) != "succeeded":
            return ExecutionResult(
                step_name="storyboard",
                success=False,
                status="failed",
                backend_name=self.name,
                backend_job_id=backend_job_id,
                error=str(latest_status.get("error") or "zimage storyboard failed"),
                metadata={"submit": submit, "status": latest_status},
            )

        output_path = str(latest_status.get("output_path") or "")
        output_url = latest_status.get("file_url")
        artifacts = []
        if output_path:
            path_obj = Path(output_path)
            artifacts.append(
                ArtifactRef(
                    key="storyboard_image",
                    kind="image",
                    path=str(path_obj),
                    origin=self.name,
                    exists=path_obj.exists(),
                    metadata={
                        "file_url": output_url,
                        "scene_id": storyboard_step.params.get("scene_id"),
                        "variation_id": storyboard_step.params.get("variation_id"),
                        "candidate_id": storyboard_step.params.get("candidate_id"),
                        "effective_prompt": effective_prompt,
                        "prompt_source": prompt_source,
                    },
                )
            )

        return ExecutionResult(
            step_name="storyboard",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=backend_job_id,
            output_path=output_path or None,
            output_url=output_url if isinstance(output_url, str) else None,
            artifacts=artifacts,
            metadata={
                "submit": submit,
                "status": latest_status,
                "effective_prompt": effective_prompt,
                "prompt_source": prompt_source,
                "candidate_prompt_text": storyboard_step.params.get("candidate_prompt_text"),
                "scene_prompt_text": storyboard_step.params.get("scene_prompt_text"),
                "storyboard_prompt_metadata": storyboard_step.params.get("storyboard_prompt_metadata"),
            },
        )

    def _resolve_effective_prompt(self, plan: ProductionPlan, storyboard_step) -> tuple[str, str]:
        step_prompt = str(storyboard_step.params.get("effective_prompt") or "").strip()
        if step_prompt:
            return step_prompt, str(storyboard_step.params.get("prompt_source") or "storyboard_step_effective_prompt")

        candidate_prompt = str(storyboard_step.params.get("candidate_prompt_text") or "").strip()
        if candidate_prompt:
            return compress_visual_prompt(candidate_prompt), "candidate_prompt_text_compressed"

        return compress_visual_prompt(plan.prompt_text), "global_plan_prompt_compressed"
