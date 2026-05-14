from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIVE_STAGE_ARTIFACTS: dict[str, str] = {
    "00": "normalized_job.json",
    "01": "pipeline_route.json",
    "02": "mode_style.json",
    "03": "skill_tree.json",
    "04": "creative_strategy.json",
    "05": "beat_hook_plan.json",
    "06": "creative_judge.json",
    "07": "scene_contracts.json",
    "08": "prompt_payload_compiled.json",
    "09": "keyframe_manifest.json",
}

LIVE_STAGE_NAMES: dict[str, str] = {
    "00": "Command Center",
    "01": "Pipeline Overview",
    "02": "Mode & Style",
    "03": "Skills laden",
    "04": "Creative Strategy",
    "05": "Beat / Hook Planner",
    "06": "Creative Judge",
    "07": "Scene Contracts",
    "08": "Prompt Compiler",
    "09": "Image / Keyframe Generation",
}

TERMINAL_STAGE_STATUSES = {"done", "error", "missing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LiveStatusWriter:
    run_dir: Path
    job_id: str

    @property
    def status_path(self) -> Path:
        return self.run_dir / "live_status.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "stage_events.jsonl"

    def initialize(self, *, viewed_stage: str = "00") -> dict[str, Any]:
        now = utc_now()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        stages = {
            stage_id: {
                "stage": stage_id,
                "name": LIVE_STAGE_NAMES[stage_id],
                "artifact": LIVE_STAGE_ARTIFACTS[stage_id],
                "status": "pending",
                "started_at": None,
                "updated_at": now,
                "finished_at": None,
                "artifact_path": str(self.run_dir / LIVE_STAGE_ARTIFACTS[stage_id]),
                "error": None,
            }
            for stage_id in LIVE_STAGE_NAMES
        }
        payload: dict[str, Any] = {
            "schema": "creative_os_live_status_v1",
            "job_id": self.job_id,
            "status": "pending",
            "viewed_stage": viewed_stage,
            "real_run_stage": "00",
            "current_running_stage": None,
            "completed_stages": [],
            "failed_stages": [],
            "pending_stages": list(LIVE_STAGE_NAMES),
            "missing_stages": [],
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "artifact_paths": {stage_id: str(self.run_dir / artifact) for stage_id, artifact in LIVE_STAGE_ARTIFACTS.items()},
            "error": None,
            "stages": stages,
        }
        self._write(payload)
        self._append_event("init", viewed_stage, "pending", None)
        return payload

    def set_viewed_stage(self, stage_id: str) -> None:
        payload = self.read()
        payload["viewed_stage"] = stage_id
        payload["updated_at"] = utc_now()
        self._write(payload)
        self._append_event("viewed", stage_id, "viewed", None)

    def stage_running(self, stage_id: str) -> None:
        payload = self.read()
        now = utc_now()
        stage = payload["stages"][stage_id]
        stage.update({"status": "running", "started_at": stage.get("started_at") or now, "updated_at": now, "error": None})
        payload.update({"status": "running", "real_run_stage": stage_id, "current_running_stage": stage_id, "updated_at": now})
        self._refresh_rollups(payload)
        self._write(payload)
        self._append_event("stage", stage_id, "running", None)

    def stage_done(self, stage_id: str, *, artifact_path: Path | None = None) -> None:
        self._finish_stage(stage_id, "done", artifact_path=artifact_path, error=None)

    def stage_missing(self, stage_id: str, *, artifact_path: Path | None = None, error: str | None = None) -> None:
        self._finish_stage(stage_id, "missing", artifact_path=artifact_path, error=error or "artifact missing")

    def stage_error(self, stage_id: str, *, artifact_path: Path | None = None, error: str | None = None) -> None:
        self._finish_stage(stage_id, "error", artifact_path=artifact_path, error=error or "unknown")

    def finish(self, *, status: str | None = None, error: str | None = None) -> None:
        payload = self.read()
        now = utc_now()
        if status is None:
            status = "error" if payload.get("failed_stages") else "complete"
        payload.update({"status": status, "current_running_stage": None, "updated_at": now, "finished_at": now, "error": error})
        self._refresh_rollups(payload)
        self._write(payload)
        self._append_event("finish", str(payload.get("real_run_stage") or "09"), status, error)

    def read(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return self.initialize()
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def _finish_stage(self, stage_id: str, status: str, *, artifact_path: Path | None, error: str | None) -> None:
        payload = self.read()
        now = utc_now()
        stage = payload["stages"][stage_id]
        if artifact_path is not None:
            stage["artifact_path"] = str(artifact_path)
            payload["artifact_paths"][stage_id] = str(artifact_path)
        stage.update({"status": status, "updated_at": now, "finished_at": now, "error": error})
        payload.update({"real_run_stage": stage_id, "current_running_stage": None, "updated_at": now})
        if status in {"error", "missing"}:
            payload["status"] = status
            payload["error"] = error
        self._refresh_rollups(payload)
        self._write(payload)
        self._append_event("stage", stage_id, status, error)

    def _refresh_rollups(self, payload: dict[str, Any]) -> None:
        stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
        payload["completed_stages"] = [stage_id for stage_id, stage in stages.items() if stage.get("status") == "done"]
        payload["failed_stages"] = [stage_id for stage_id, stage in stages.items() if stage.get("status") == "error"]
        payload["missing_stages"] = [stage_id for stage_id, stage in stages.items() if stage.get("status") == "missing"]
        payload["pending_stages"] = [stage_id for stage_id, stage in stages.items() if stage.get("status") == "pending"]

    def _write(self, payload: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _append_event(self, kind: str, stage_id: str, status: str, error: str | None) -> None:
        event = {"ts": utc_now(), "kind": kind, "stage": stage_id, "status": status, "error": error}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
