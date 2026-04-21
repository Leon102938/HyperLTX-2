from __future__ import annotations

import time
from pathlib import Path

from agent_core.adapters.base import MusicBackendAdapter
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.utils import http_json, probe_media_duration


class MusicAdapter(MusicBackendAdapter):
    name = "ace_step_1_5_music"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        poll_interval_sec: float = 3.0,
        max_wait_sec: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval_sec = poll_interval_sec
        self.max_wait_sec = max_wait_sec

    def capabilities(self) -> BackendCapabilities:
        try:
            payload = http_json(f"{self.base_url}/DW/ace_step_1_5_ready")
            available = bool(payload.get("ready"))
        except Exception as exc:
            available = False
            note = f"readiness probe failed: {exc}"
        else:
            note = "uses local FastAPI /Ace_step_1.5 endpoints for instrumental music generation"

        return BackendCapabilities(
            name=self.name,
            kind="music",
            available=available,
            phase1_enabled=available,
            transport="http",
            supported_pipelines=["text2music"],
            supported_orientations=["landscape", "portrait", "square"],
            supported_resolution_labels=["draft", "standard", "high", "custom"],
            notes=[note],
        )

    def generate_music(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        duration_sec = round(max(4.0, float(job.metadata.get("music_duration_sec", plan.target_duration_sec))), 2)
        prompt = self._build_music_prompt(job)
        payload = {
            "job_id": f"{job.job_id}_music",
            "caption": prompt,
            "global_caption": prompt,
            "duration": duration_sec,
            "instrumental": True,
            "audio_format": str(job.metadata.get("music_audio_format", "flac")),
            "inference_steps": int(job.metadata.get("music_inference_steps", 8)),
            "guidance_scale": float(job.metadata.get("music_guidance_scale", 7.0)),
            "fade_in_duration": float(job.metadata.get("music_fade_in_duration", 0.35)),
            "fade_out_duration": float(job.metadata.get("music_fade_out_duration", 1.2)),
            "normalization_db": float(job.metadata.get("music_normalization_db", -14.0)),
            "thinking": bool(job.metadata.get("music_use_cot", True)),
            "cot_duration": duration_sec,
            "cot_caption": prompt,
        }

        submit = http_json(f"{self.base_url}/Ace_step_1.5/generate", method="POST", payload=payload, timeout=60)
        backend_job_id = str(submit["job_id"])
        status_url = f"{self.base_url}/Ace_step_1.5/status/{backend_job_id}"
        get_url = f"{self.base_url}/Ace_step_1.5/get/{backend_job_id}"

        deadline = time.monotonic() + self.max_wait_sec
        latest_status: dict[str, object] = {}
        while time.monotonic() < deadline:
            latest_status = http_json(status_url, timeout=60)
            status = str(latest_status.get("status") or latest_status.get("state"))
            if status in {"succeeded", "failed"}:
                break
            time.sleep(self.poll_interval_sec)
        else:
            raise RuntimeError(f"ACE-Step music job {backend_job_id} timed out after {self.max_wait_sec}s")

        result_payload = http_json(get_url, timeout=60)
        if not result_payload.get("ok"):
            error = str(result_payload.get("error") or latest_status.get("error") or "ace-step music failed")
            return ExecutionResult(
                step_name="music",
                success=False,
                status="failed",
                backend_name=self.name,
                backend_job_id=backend_job_id,
                error=error,
                metadata={"submit": submit, "status": latest_status, "result": result_payload},
            )

        output_path = str(result_payload.get("primary_output_path") or "")
        duration = probe_media_duration(output_path)
        artifacts: list[ArtifactRef] = []
        if output_path:
            path_obj = Path(output_path)
            artifacts.append(
                ArtifactRef(
                    key="background_music",
                    kind="audio",
                    path=str(path_obj),
                    origin=self.name,
                    exists=path_obj.exists(),
                    metadata={
                        "file_url": result_payload.get("file_url"),
                        "requested_duration_sec": duration_sec,
                        "voice_present": bool(voice_result and voice_result.output_path),
                        "prompt": prompt,
                    },
                )
            )

        return ExecutionResult(
            step_name="music",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=backend_job_id,
            output_path=output_path or None,
            output_url=result_payload.get("file_url"),
            duration_sec=duration,
            artifacts=artifacts,
            metadata={"submit": submit, "status": latest_status, "result": result_payload, "prompt": prompt},
        )

    @staticmethod
    def _build_music_prompt(job: JobInput) -> str:
        custom_prompt = str(job.metadata.get("music_prompt") or "").strip()
        if custom_prompt:
            return custom_prompt

        concept = " ".join((job.idea or job.script or "short-form social explainer").split())[:220]
        style = " ".join((job.style or "cinematic").split())[:80]
        return (
            f"Instrumental background music for a {style} short-form content video about {concept}. "
            "Keep it modern, motivating, clean, supportive for voiceover, no vocals, no harsh drops, no dominant lead."
        )
