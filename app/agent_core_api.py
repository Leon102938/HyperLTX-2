from __future__ import annotations

import threading
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from agent_core.agent import VideoAgent
from agent_core.schemas import JobInput, ResultSummary
from agent_core.state_store import StateStore


router = APIRouter(prefix="/agent-core", tags=["agent-core"])


class AgentCoreRunRequest(BaseModel):
    job: JobInput


class BackgroundJobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def register_accepted(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {"status": "accepted", "current_phase": "received"}

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.setdefault(job_id, {})
            record.update({"status": "running"})

    def mark_finished(self, job_id: str, *, success: bool) -> None:
        with self._lock:
            record = self._jobs.setdefault(job_id, {})
            record.update({"status": "done" if success else "failed"})

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def submit(self, *, job: JobInput, agent: VideoAgent) -> None:
        self.register_accepted(job.job_id or "")

        def _run() -> None:
            self.mark_running(job.job_id or "")
            result = agent.run_job(job, raise_on_error=False)
            self.mark_finished(job.job_id or "", success=result.success)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


_job_runner = BackgroundJobRunner()


def get_state_store() -> StateStore:
    return StateStore()


def get_video_agent() -> VideoAgent:
    return VideoAgent()


def get_job_runner() -> BackgroundJobRunner:
    return _job_runner


def _read_json_if_ready(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return __import__("json").loads(path.read_text())
    except (OSError, JSONDecodeError):
        return None


def _build_refs(job_id: str, store: StateStore) -> dict[str, str]:
    job_dir = store.job_dir(job_id)
    return {
        "job_dir": str(job_dir),
        "state_json_path": str(job_dir / "state.json"),
        "result_json_path": str(job_dir / "result.json"),
        "final_mp4_path": str(job_dir / "final.mp4"),
        "state_json_url": f"/agent-runs/{job_id}/state.json",
        "result_json_url": f"/agent-runs/{job_id}/result.json",
        "final_mp4_url": f"/agent-runs/{job_id}/final.mp4",
    }


def _build_public_refs(refs: dict[str, str]) -> dict[str, str]:
    return {
        "state_json_url": refs["state_json_url"],
        "result_json_url": refs["result_json_url"],
        "final_mp4_url": refs["final_mp4_url"],
    }


def _sync_success_payload(job_id: str, result: ResultSummary, store: StateStore) -> dict[str, Any]:
    refs = _build_refs(job_id, store)
    final_ready = bool(result.output_final_path and Path(result.output_final_path).exists())
    result_ready = Path(refs["result_json_path"]).exists()
    payload = {
        "ok": True,
        "job_id": job_id,
        "status": result.final_phase,
        "current_phase": result.final_phase,
        "success": result.success,
        "result": result.model_dump(mode="json"),
        "error": None,
        "poll_url": f"/agent-core/jobs/{job_id}",
        "status_summary": "All final artifacts are ready." if final_ready and result_ready else "Result completed.",
        "is_terminal": True,
        "should_poll": False,
        "retry_after_sec": None,
        "artifacts_ready": result_ready,
        "final_mp4_ready": final_ready,
        "result_json_ready": result_ready,
        "refs": refs,
        "public_refs": _build_public_refs(refs),
    }
    return payload


def _status_payload(job_id: str, store: StateStore, runner: BackgroundJobRunner) -> dict[str, Any]:
    refs = _build_refs(job_id, store)
    state_payload = _read_json_if_ready(Path(refs["state_json_path"]))
    result_payload = _read_json_if_ready(Path(refs["result_json_path"]))
    runner_payload = runner.get(job_id)

    if state_payload is None and result_payload is None and runner_payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")

    if result_payload is not None:
        success = bool(result_payload.get("success"))
        final_ready = success and Path(refs["final_mp4_path"]).exists()
        return {
            "ok": True,
            "job_id": job_id,
            "status": "done" if success else "failed",
            "current_phase": (state_payload or {}).get("current_phase", result_payload.get("final_phase")),
            "success": success,
            "result": result_payload,
            "error": None if success else {"type": "agent_core_job_failed", "message": result_payload.get("message")},
            "poll_url": f"/agent-core/jobs/{job_id}",
            "status_summary": "All final artifacts are ready." if success else "Failure result is ready.",
            "is_terminal": True,
            "should_poll": False,
            "retry_after_sec": None,
            "artifacts_ready": True,
            "final_mp4_ready": bool(final_ready),
            "result_json_ready": True,
            "refs": refs,
            "public_refs": _build_public_refs(refs),
        }

    current_phase = (state_payload or {}).get("current_phase", (runner_payload or {}).get("current_phase", "received"))
    accepted = current_phase == "received" and (state_payload is None or not (state_payload.get("steps") or {}))
    return {
        "ok": True,
        "job_id": job_id,
        "status": "accepted" if accepted else "running",
        "current_phase": current_phase,
        "success": None,
        "result": None,
        "error": None,
        "poll_url": f"/agent-core/jobs/{job_id}",
        "status_summary": "Job accepted and waiting for work." if accepted else "Job is still running.",
        "is_terminal": False,
        "should_poll": True,
        "retry_after_sec": 2 if accepted else 3,
        "artifacts_ready": False,
        "final_mp4_ready": False,
        "result_json_ready": False,
        "refs": refs,
        "public_refs": _build_public_refs(refs),
    }


@router.post("/run")
def run_job(
    request: AgentCoreRunRequest,
    agent: VideoAgent = Depends(get_video_agent),
    store: StateStore = Depends(get_state_store),
) -> dict[str, Any]:
    result = agent.run_job(request.job, raise_on_error=False)
    return _sync_success_payload(request.job.job_id or result.job_id, result, store)


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    request: AgentCoreRunRequest,
    agent: VideoAgent = Depends(get_video_agent),
    runner: BackgroundJobRunner = Depends(get_job_runner),
) -> dict[str, Any]:
    job = agent.load_job(request.job)
    runner.submit(job=job, agent=agent)
    return {
        "ok": True,
        "job_id": job.job_id,
        "status": "accepted",
        "current_phase": "received",
        "success": None,
        "result": None,
        "error": None,
        "poll_url": f"/agent-core/jobs/{job.job_id}",
        "status_summary": "Job accepted and waiting for work.",
        "is_terminal": False,
        "should_poll": True,
        "retry_after_sec": 2,
        "artifacts_ready": False,
        "final_mp4_ready": False,
        "result_json_ready": False,
        "refs": _build_refs(job.job_id or "", StateStore()),
        "public_refs": _build_public_refs(_build_refs(job.job_id or "", StateStore())),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, store: StateStore = Depends(get_state_store), runner: BackgroundJobRunner = Depends(get_job_runner)) -> dict[str, Any]:
    return _status_payload(job_id, store, runner)
