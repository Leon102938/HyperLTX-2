from __future__ import annotations

import time
from pathlib import Path

from agent_core.adapters.base import VoiceAdapter
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.utils import http_json, probe_media_duration


class QwenTTSAdapter(VoiceAdapter):
    name = "qwen_tts"

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
        try:
            payload = http_json(f"{self.base_url}/DW/qwen_tts_ready")
            available = bool(payload.get("ready"))
        except Exception as exc:
            available = False
            note = f"readiness probe failed: {exc}"
        else:
            note = "uses local FastAPI /qwen_tts endpoints"

        return BackendCapabilities(
            name=self.name,
            kind="voice",
            available=available,
            phase1_enabled=True,
            transport="http",
            supported_pipelines=["custom_voice"],
            supported_orientations=["landscape", "portrait", "square"],
            supported_resolution_labels=["draft", "standard", "high", "custom"],
            notes=[note],
        )

    def generate_voice(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        payload = {
            "job_id": f"{job.job_id}_voice",
            "text": job.script or job.idea,
            "speaker": job.voice_id or "Ryan",
            "language": str(job.metadata.get("language", "German")),
            "instruct": job.extra_llm_instruction or f"{job.style} narration voiceover",
        }
        submit = http_json(f"{self.base_url}/qwen_tts/custom_voice", method="POST", payload=payload)
        backend_job_id = str(submit["job_id"])
        get_url = f"{self.base_url}/qwen_tts/get/{backend_job_id}"
        status_url = f"{self.base_url}/qwen_tts/status/{backend_job_id}"

        deadline = time.monotonic() + self.max_wait_sec
        latest_status: dict[str, object] = {}
        while time.monotonic() < deadline:
            latest_status = http_json(status_url)
            status = str(latest_status.get("status") or latest_status.get("state"))
            if status in {"succeeded", "failed"}:
                break
            time.sleep(self.poll_interval_sec)
        else:
            raise RuntimeError(f"Qwen TTS job {backend_job_id} timed out after {self.max_wait_sec}s")

        result_payload = http_json(get_url)
        if not result_payload.get("ok"):
            error = str(result_payload.get("error") or latest_status.get("error") or "qwen tts failed")
            return ExecutionResult(
                step_name="voice",
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
                    key="voice_audio",
                    kind="audio",
                    path=str(path_obj),
                    origin=self.name,
                    exists=path_obj.exists(),
                    metadata={
                        "file_url": result_payload.get("file_url"),
                        "sample_rate": result_payload.get("sample_rate"),
                    },
                )
            )

        return ExecutionResult(
            step_name="voice",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=backend_job_id,
            output_path=output_path or None,
            output_url=result_payload.get("file_url"),
            duration_sec=duration_sec,
            artifacts=artifacts,
            metadata={"submit": submit, "status": latest_status, "result": result_payload},
        )
