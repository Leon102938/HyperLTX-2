from __future__ import annotations

from pathlib import Path

from agent_core.schemas import ArtifactRef, ExecutionResult, JobInput, JobPhase, JobState, ProductionPlan, ResultSummary, StepRunRecord
from agent_core.utils import ensure_dir, utc_now_iso, write_json


class StateStore:
    def __init__(self, root_dir: str | Path = "/workspace/agent_runs") -> None:
        self.root_dir = ensure_dir(root_dir)

    def job_dir(self, job_id: str) -> Path:
        return self.root_dir / job_id

    def path_for(self, job_id: str, name: str) -> Path:
        mapping = {
            "input": self.job_dir(job_id) / "input_job.json",
            "plan": self.job_dir(job_id) / "plan.json",
            "scene_plan": self.job_dir(job_id) / "scene_plan.json",
            "takes": self.job_dir(job_id) / "takes.json",
            "state": self.job_dir(job_id) / "state.json",
            "result": self.job_dir(job_id) / "result.json",
            "log": self.job_dir(job_id) / "logs" / "agent.log",
        }
        return mapping[name]

    def initialize(self, job: JobInput) -> JobState:
        job_dir = self.job_dir(job.job_id or "")
        ensure_dir(job_dir)
        ensure_dir(job_dir / "logs")
        write_json(self.path_for(job.job_id or "", "input"), job.model_dump(mode="json"))

        now = utc_now_iso()
        state = JobState(
            job_id=job.job_id or "",
            status="received",
            current_phase="received",
            created_at=now,
            updated_at=now,
            artifacts=[
                ArtifactRef(
                    key="input_job",
                    kind="json",
                    path=str(self.path_for(job.job_id or "", "input")),
                    origin="agent_core",
                    exists=True,
                ),
                ArtifactRef(
                    key="state_file",
                    kind="json",
                    path=str(self.path_for(job.job_id or "", "state")),
                    origin="agent_core",
                    exists=False,
                ),
                ArtifactRef(
                    key="agent_log",
                    kind="log",
                    path=str(self.path_for(job.job_id or "", "log")),
                    origin="agent_core",
                    exists=True,
                ),
            ],
        )
        self.append_log(job.job_id or "", "job received")
        self.save_state(state)
        return state

    def save_state(self, state: JobState) -> Path:
        state.updated_at = utc_now_iso()
        path = write_json(self.path_for(state.job_id, "state"), state.model_dump(mode="json"))
        self._upsert_artifact(
            state,
            ArtifactRef(
                key="state_file",
                kind="json",
                path=str(path),
                origin="agent_core",
                exists=True,
            ),
        )
        write_json(path, state.model_dump(mode="json"))
        return path

    def save_plan(self, state: JobState, plan: ProductionPlan) -> Path:
        state.plan_version += 1
        path = write_json(self.path_for(state.job_id, "plan"), plan.model_dump(mode="json"))
        self._upsert_artifact(
            state,
            ArtifactRef(
                key="plan_file",
                kind="json",
                path=str(path),
                origin="agent_core",
                exists=True,
                metadata={"plan_version": state.plan_version},
            ),
        )
        for step in plan.steps:
            existing = state.steps.get(step.name)
            if existing is None:
                state.steps[step.name] = StepRunRecord(
                    name=step.name,
                    status="planned" if step.enabled else "skipped",
                    details={"skip_reason": step.skip_reason, "params": step.params},
                )
            else:
                existing.details = {"skip_reason": step.skip_reason, "params": step.params}
                if existing.status in {"planned", "skipped"}:
                    existing.status = "planned" if step.enabled else "skipped"
                state.steps[step.name] = existing
        self.save_state(state)
        return path

    def save_result(self, state: JobState, result: ResultSummary) -> Path:
        path = write_json(self.path_for(state.job_id, "result"), result.model_dump(mode="json"))
        state.result_path = str(path)
        self._upsert_artifact(
            state,
            ArtifactRef(
                key="result_file",
                kind="json",
                path=str(path),
                origin="agent_core",
                exists=True,
            ),
        )
        self.save_state(state)
        return path

    def save_scene_plan(self, state: JobState, plan: ProductionPlan) -> Path:
        payload = {
            "job_id": plan.job_id,
            "scene_count": len(plan.scenes),
            "segmentation_mode": plan.metadata.get("segmentation_mode", "single_scene"),
            "target_duration_sec": plan.target_duration_sec,
            "scenes": [scene.model_dump(mode="json") for scene in plan.scenes],
        }
        path = write_json(self.path_for(state.job_id, "scene_plan"), payload)
        self._upsert_artifact(
            state,
            ArtifactRef(
                key="scene_plan_file",
                kind="json",
                path=str(path),
                origin="agent_core",
                exists=True,
                metadata={"scene_count": len(plan.scenes)},
            ),
        )
        self.save_state(state)
        return path

    def save_take_report(self, state: JobState, payload: dict[str, object]) -> Path:
        path = write_json(self.path_for(state.job_id, "takes"), payload)
        self._upsert_artifact(
            state,
            ArtifactRef(
                key="take_report_file",
                kind="json",
                path=str(path),
                origin="agent_core",
                exists=True,
                metadata={
                    "scene_count": payload.get("scene_count"),
                    "selection_mode": payload.get("selection_mode"),
                },
            ),
        )
        self.save_state(state)
        return path

    def transition(self, state: JobState, phase: JobPhase, note: str | None = None) -> JobState:
        state.status = phase
        state.current_phase = phase
        if note:
            state.notes.append(note)
            self.append_log(state.job_id, note)
        self.save_state(state)
        return state

    def start_step(self, state: JobState, step_name: str, backend_name: str) -> None:
        record = state.steps.get(step_name) or StepRunRecord(name=step_name)
        record.status = "running"
        record.started_at = utc_now_iso()
        record.backend_name = backend_name
        state.steps[step_name] = record
        self.append_log(state.job_id, f"step {step_name} started via {backend_name}")
        self.save_state(state)

    def finish_step(self, state: JobState, result: ExecutionResult) -> None:
        record = state.steps.get(result.step_name) or StepRunRecord(name=result.step_name)
        record.status = result.status
        record.finished_at = utc_now_iso()
        record.backend_name = result.backend_name
        record.backend_job_id = result.backend_job_id
        record.output_path = result.output_path
        record.output_url = result.output_url
        record.duration_sec = result.duration_sec
        record.error = result.error
        record.details = result.metadata
        state.steps[result.step_name] = record
        for artifact in result.artifacts:
            self._upsert_artifact(state, artifact)
        message = f"step {result.step_name} finished with status={result.status}"
        if result.error:
            message += f" error={result.error}"
        self.append_log(state.job_id, message)
        self.save_state(state)

    def fail(self, state: JobState, message: str) -> JobState:
        state.errors.append(message)
        self.append_log(state.job_id, f"job failed: {message}")
        return self.transition(state, "failed")

    def append_log(self, job_id: str, message: str) -> Path:
        log_path = self.path_for(job_id, "log")
        ensure_dir(log_path.parent)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now_iso()}] {message}\n")
        return log_path

    def _upsert_artifact(self, state: JobState, artifact: ArtifactRef) -> None:
        for index, existing in enumerate(state.artifacts):
            if existing.key == artifact.key:
                state.artifacts[index] = artifact
                return
        state.artifacts.append(artifact)
