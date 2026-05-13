from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_core.creative_os.run_inspector import CreativeOSRunInspector, RunInspection

FIXTURE_MARKER = "tests/fixtures/creative_os_runs"


@dataclass(frozen=True)
class HeaderData:
    job_id: str
    pipeline: str
    mode: str
    topic: str
    orientation: str
    resolution: str
    duration: int
    scene_count: int
    status: str
    session: str
    checks: str
    render_state: str
    watch: str
    artifact_mode: str
    run_type: str


@dataclass(frozen=True)
class SystemStatusData:
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PipelineStageData:
    index: str
    name: str
    status: str


@dataclass(frozen=True)
class WorkspaceSceneData:
    scene_id: str
    keyframe: str
    summary: str
    state_label: str = "READY FOR LTX"
    status: str = "waiting for Stage 09 render gate"
    backend: str = "not_checked"
    backend_status: str = "not_checked"
    overall_status: str = "not_checked"
    progress_percent: str = ""
    elapsed: str = ""
    output_path: str = ""
    error: str = ""
    backend_job_id: str = ""
    file_exists: bool = False
    file_size_bytes: int = 0
    file_mtime: str = ""
    gallery_path: str = ""


@dataclass(frozen=True)
class WorkspaceData:
    current_step: str
    last_passed: str
    next_technical: str
    operator_focus: str
    render_paused: str
    scenes: tuple[WorkspaceSceneData, ...]


@dataclass(frozen=True)
class SkillHealthData:
    mark: str
    status: str
    loaded_count: int
    fallback_count: int
    missing_optional_count: int
    blocking_missing_count: int


@dataclass(frozen=True)
class ArtifactsData:
    lines: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class IssuesData:
    blocking_issues: tuple[str, ...]
    severity: str


@dataclass(frozen=True)
class NextData:
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CockpitState:
    inspection: RunInspection
    session_mode: str
    run_type: str
    data_source_path: Path
    run_found: bool
    last_refresh_time: str
    watch_enabled: bool
    refresh_sec: float
    header: HeaderData
    system_status: SystemStatusData
    pipeline_map: tuple[PipelineStageData, ...]
    workspace: WorkspaceData
    skill_health: SkillHealthData
    artifacts: ArtifactsData
    issues: IssuesData
    next_panel: NextData
    selected_stage: str = "09"
    selected_image_job: int = 2
    expanded_image_jobs: tuple[int, ...] = (2,)


class CockpitStateAdapter:
    def __init__(self, *, job_id: str, runs_root: str | Path, watch_enabled: bool = False, refresh_sec: float = 2.0) -> None:
        self.job_id = job_id
        self.runs_root = Path(runs_root)
        self.watch_enabled = watch_enabled
        self.refresh_sec = refresh_sec
        self.inspector = CreativeOSRunInspector(runs_root=self.runs_root)

    def load(self) -> CockpitState:
        inspection = self.inspector.inspect(self.job_id)
        data_source_path = self.runs_root / self.job_id
        run_found = data_source_path.exists()
        run_type = _run_type(data_source_path)
        session_mode = _session_mode(data_source_path, run_found)
        refresh_time = datetime.now().strftime("%H:%M:%S")
        if not run_found:
            return self._missing_state(
                inspection=inspection,
                data_source_path=data_source_path,
                session_mode=session_mode,
                refresh_time=refresh_time,
            )
        if run_type == "agent_core":
            return self._agent_core_state(
                inspection=inspection,
                data_source_path=data_source_path,
                session_mode=session_mode,
                refresh_time=refresh_time,
            )
        if run_type == "unknown":
            return self._unknown_state(
                inspection=inspection,
                data_source_path=data_source_path,
                session_mode=session_mode,
                refresh_time=refresh_time,
            )
        meta = _job_meta(inspection)
        health = self.inspector.skill_health(inspection)
        workspace_scenes = tuple(_motion_items(inspection))
        return CockpitState(
            inspection=inspection,
            session_mode=session_mode,
            run_type=run_type,
            data_source_path=data_source_path,
            run_found=run_found,
            last_refresh_time=refresh_time,
            watch_enabled=self.watch_enabled,
            refresh_sec=self.refresh_sec,
            header=HeaderData(
                job_id=str(inspection.job_id),
                pipeline="shortform_storyboard_v1",
                mode=str(meta["mode"]),
                topic=str(meta["topic"]),
                orientation=str(meta["orientation"]),
                resolution=str(meta["resolution"]),
                duration=int(meta["duration"]),
                scene_count=int(meta["scene_count"]),
                status=inspection.status,
                session=_session_label(session_mode),
                checks="no live checks",
                render_state="paused",
                watch=_watch_label(self.watch_enabled, self.refresh_sec),
                artifact_mode="fixture" if session_mode == "fixture/demo" else "artifact read",
                run_type=run_type,
            ),
            system_status=SystemStatusData(rows=_system_rows(inspection)),
            pipeline_map=tuple(PipelineStageData(stage.index, stage.name, stage.status) for stage in inspection.stages),
            workspace=WorkspaceData(
                current_step=_creative_os_current_step(inspection, session_mode),
                last_passed=_creative_os_last_completed(inspection, self.inspector.last_passed_stage(inspection)),
                next_technical=_creative_os_next_available(inspection),
                operator_focus=_creative_os_operator_focus(inspection, session_mode),
                render_paused="yes",
                scenes=workspace_scenes,
            ),
            skill_health=SkillHealthData(
                mark=str(health["mark"]),
                status=str(health["status"]),
                loaded_count=len(health["loaded"]),
                fallback_count=len(health["fallbacks"]),
                missing_optional_count=len(health["missing_optional"]),
                blocking_missing_count=len(health["blocking_missing"]),
            ),
            artifacts=ArtifactsData(
                lines=(
                    (f"{_artifact_mark(inspection, 'zimage_prompts.json')} zimage_prompts.json", bool(inspection.artifacts.get("zimage_prompts.json"))),
                    _keyframe_artifact_line(inspection, workspace_scenes),
                    (f"{_artifact_mark(inspection, 'keyframe_gallery.html')} keyframe_gallery.html", bool(inspection.artifacts.get("keyframe_gallery.html"))),
                    (
                        f"{_artifact_mark(inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
                        bool(inspection.artifacts.get("ltx_motion_prompts.json")),
                    ),
                    (
                        f"{_artifact_mark(inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
                        bool(inspection.artifacts.get("ltx_prompt_audit.json")),
                    ),
                    (
                        f"{_artifact_mark(inspection, 'ltx_video_takes_manifest.json')} video_takes_manifest",
                        bool(inspection.artifacts.get("ltx_video_takes_manifest.json")),
                    ),
                )
            ),
            issues=IssuesData(
                blocking_issues=tuple(str(issue) for issue in inspection.blocking_issues),
                severity=_issues_severity(tuple(str(issue) for issue in inspection.blocking_issues)),
            ),
            next_panel=NextData(rows=_creative_os_next_rows(inspection)),
        )

    def _agent_core_state(
        self,
        *,
        inspection: RunInspection,
        data_source_path: Path,
        session_mode: str,
        refresh_time: str,
    ) -> CockpitState:
        data = _agent_core_data(data_source_path)
        result = data.get("result.json") or {}
        state = data.get("state.json") or {}
        plan = data.get("plan.json") or {}
        scene_plan = data.get("scene_plan.json") or {}
        director = data.get("director_output.json") or {}
        checkpoints = data.get("checkpoints.json") or {}

        director_mode = str(director.get("director_mode") or _nested(plan, "director_output", "mode") or "unknown")
        director_llm_active = bool(director.get("director_llm_active") or _nested(plan, "director_output", "llm_active"))
        final_mp4 = (data_source_path / "final.mp4").exists()
        scene_count = int(scene_plan.get("scene_count") or len(scene_plan.get("scenes") or []) or len(plan.get("scenes") or []) or 0)
        current_step = str(
            state.get("current_phase")
            or state.get("status")
            or result.get("final_phase")
            or checkpoints.get("current_checkpoint_id")
            or "unknown"
        )
        status = str(result.get("final_phase") or state.get("status") or inspection.status)
        issues = _agent_core_issues(director_mode, director_llm_active, final_mp4)
        artifacts = _agent_core_artifacts(data_source_path)
        scenes = _agent_core_scenes(scene_plan, plan)
        return CockpitState(
            inspection=inspection,
            session_mode=session_mode,
            run_type="agent_core",
            data_source_path=data_source_path,
            run_found=True,
            last_refresh_time=refresh_time,
            watch_enabled=self.watch_enabled,
            refresh_sec=self.refresh_sec,
            header=HeaderData(
                job_id=self.job_id,
                pipeline=str(plan.get("selected_pipeline") or state.get("pipeline_id") or "agent_core"),
                mode=director_mode,
                topic=str(_nested(director, "director_output", "creative_brief", "concept") or plan.get("prompt_text") or "agent-core run"),
                orientation=str(plan.get("orientation") or "unknown"),
                resolution=_agent_core_resolution(plan),
                duration=int(float(plan.get("target_duration_sec") or result.get("planned_duration_sec") or 0)),
                scene_count=scene_count,
                status=status,
                session=_session_label(session_mode),
                checks="no live checks",
                render_state="paused" if not final_mp4 else "complete",
                watch=_watch_label(self.watch_enabled, self.refresh_sec),
                artifact_mode="artifact read",
                run_type="agent_core",
            ),
            system_status=SystemStatusData(
                rows=(
                    ("API", "? not_checked"),
                    ("Director", _director_status(director_mode, director_llm_active)),
                    ("Director LLM", "✓ active" if director_llm_active else "✗ inactive"),
                    ("Final MP4", "✓ present" if final_mp4 else "○ missing"),
                    ("Run Type", "agent_core"),
                    ("Mode", "read only"),
                    ("Subtitles", "- off"),
                )
            ),
            pipeline_map=_agent_core_pipeline_map(checkpoints, state),
            workspace=WorkspaceData(
                current_step=current_step,
                last_passed=_agent_core_last_passed(checkpoints),
                next_technical=_agent_core_next(issues, final_mp4),
                operator_focus=f"Director {director_mode}; llm_active={str(director_llm_active).lower()}",
                render_paused="yes" if not final_mp4 else "no",
                scenes=scenes,
            ),
            skill_health=SkillHealthData(
                mark="✓",
                status="ok",
                loaded_count=0,
                fallback_count=0,
                missing_optional_count=0,
                blocking_missing_count=0,
            ),
            artifacts=ArtifactsData(lines=artifacts),
            issues=IssuesData(blocking_issues=issues, severity=_issues_severity(issues)),
            next_panel=NextData(rows=_agent_core_next_rows(issues, final_mp4)),
        )

    def _missing_state(
        self,
        *,
        inspection: RunInspection,
        data_source_path: Path,
        session_mode: str,
        refresh_time: str,
    ) -> CockpitState:
        return CockpitState(
            inspection=inspection,
            session_mode=session_mode,
            run_type="missing",
            data_source_path=data_source_path,
            run_found=False,
            last_refresh_time=refresh_time,
            watch_enabled=self.watch_enabled,
            refresh_sec=self.refresh_sec,
            header=HeaderData(
                job_id=self.job_id,
                pipeline="shortform_storyboard_v1",
                mode="unknown",
                topic="unknown",
                orientation="unknown",
                resolution="unknown",
                duration=0,
                scene_count=0,
                status="run_not_found",
                session="missing",
                checks="no live checks",
                render_state="unknown",
                watch=_watch_label(self.watch_enabled, self.refresh_sec),
                artifact_mode="read only",
                run_type="missing",
            ),
            system_status=SystemStatusData(
                rows=(
                    ("Run", "not found"),
                    ("Searched", str(data_source_path)),
                    ("Hint", "use --runs-root for fixture/demo data"),
                    ("Mode", "read only"),
                )
            ),
            pipeline_map=tuple(),
            workspace=WorkspaceData(
                current_step="Run not found",
                last_passed="- none",
                next_technical="create a real run first",
                operator_focus="check job id or --runs-root",
                render_paused="unknown",
                scenes=tuple(),
            ),
            skill_health=SkillHealthData(
                mark="?",
                status="unknown",
                loaded_count=0,
                fallback_count=0,
                missing_optional_count=0,
                blocking_missing_count=0,
            ),
            artifacts=ArtifactsData(lines=(("○ run artifacts unavailable", False),)),
            issues=IssuesData(blocking_issues=("Run not found", f"searched: {data_source_path}"), severity="error"),
            next_panel=NextData(rows=(("Hint", "use --runs-root for fixture/demo data"), ("Real run", "create a run before watching"))),
        )

    def _unknown_state(
        self,
        *,
        inspection: RunInspection,
        data_source_path: Path,
        session_mode: str,
        refresh_time: str,
    ) -> CockpitState:
        state = self._missing_state(
            inspection=inspection,
            data_source_path=data_source_path,
            session_mode=session_mode,
            refresh_time=refresh_time,
        )
        return CockpitState(
            inspection=state.inspection,
            session_mode=state.session_mode,
            run_type="unknown",
            data_source_path=state.data_source_path,
            run_found=True,
            last_refresh_time=state.last_refresh_time,
            watch_enabled=state.watch_enabled,
            refresh_sec=state.refresh_sec,
            header=HeaderData(
                **{**state.header.__dict__, "session": _session_label(session_mode), "run_type": "unknown", "status": "unknown_run_type"}
            ),
            system_status=SystemStatusData(rows=(("Run", "unknown type"), ("Searched", str(data_source_path)), ("Mode", "read only"))),
            pipeline_map=tuple(),
            workspace=WorkspaceData(
                current_step="Unknown run type",
                last_passed="- none",
                next_technical="inspect run artifacts",
                operator_focus="known Agent-Core/Creative-OS files missing",
                render_paused="unknown",
                scenes=tuple(),
            ),
            skill_health=state.skill_health,
            artifacts=ArtifactsData(lines=(("○ known run artifacts unavailable", False),)),
            issues=IssuesData(blocking_issues=("Unknown run type", f"searched: {data_source_path}"), severity="error"),
            next_panel=NextData(rows=(("Technical", "inspect run artifacts"), ("Mode", "read only"))),
        )


def _system_rows(inspection: RunInspection) -> tuple[tuple[str, str], ...]:
    vision = "- manual_structured" if "stage6_review_decision" in inspection.data else "- heuristic"
    image = _image_backend_row(inspection)
    return (
        ("API", "? not_checked"),
        ("Director", "? not_checked"),
        ("Image Backend", image),
        ("Video Backend", "? planned ltx2"),
        ("Vision Review", vision),
        ("Voice", "- disabled"),
        ("Music", "- disabled"),
        ("Subtitles", "- off"),
    )


def _image_backend_row(inspection: RunInspection) -> str:
    manifest = inspection.data.get("keyframe_manifest")
    if isinstance(manifest, dict):
        backend = str(manifest.get("backend") or "zimage_http")
        if manifest.get("backend_status") == "available":
            return f"✓ {backend}"
        reason = str(manifest.get("backend_reason") or "missing")
        return f"✗ {backend} · {reason}"
    return "? not_checked"


def _job_meta(inspection: RunInspection) -> dict[str, object]:
    job = inspection.data.get("normalized_job") or {}
    route = inspection.data.get("intent_route") or {}
    scenes = inspection.data.get("scene_contracts") or []
    return {
        "resolution": job.get("resolution") or "512x768",
        "duration": job.get("duration_sec") or 9,
        "orientation": job.get("orientation") or "portrait",
        "mode": route.get("genre_intent") or route.get("mode_intent") or "unknown",
        "topic": route.get("topic_intent") or "unknown",
        "scene_count": len(scenes) or 3,
    }


def _creative_os_current_step(inspection: RunInspection, session_mode: str) -> str:
    phase1 = inspection.data.get("phase1_status")
    if isinstance(phase1, dict):
        return f"Stage {phase1.get('real_run_stage') or phase1.get('current_stage') or '09'} Image / Keyframe Generation"
    return "LTX Motion Ready" if session_mode == "fixture/demo" else "not_checked"


def _creative_os_last_completed(inspection: RunInspection, fallback: str) -> str:
    phase1 = inspection.data.get("phase1_status")
    if isinstance(phase1, dict):
        value = phase1.get("last_completed_stage")
        return f"✓ {value}" if value else "- none"
    return f"✓ {fallback}"


def _creative_os_next_available(inspection: RunInspection) -> str:
    phase1 = inspection.data.get("phase1_status")
    if isinstance(phase1, dict):
        if phase1.get("next_available_stage") == "none_phase1_complete":
            return "Phase 1 complete / Stage 10+ not built yet"
        if phase1.get("paused_reason"):
            return "Stage 09 paused / image backend unavailable"
        return f"Stage {phase1.get('next_available_stage') or '09'}"
    return "○ 09 LTX I2V takes"


def _creative_os_operator_focus(inspection: RunInspection, session_mode: str) -> str:
    if session_mode == "fixture/demo":
        return "fixture/demo artifacts"
    phase1 = inspection.data.get("phase1_status")
    if isinstance(phase1, dict):
        return "Source: real_run · Fixture: no · Artifacts: loaded"
    return "Source: real_run · Fixture: no · Artifacts: partial"


def _creative_os_next_rows(inspection: RunInspection) -> tuple[tuple[str, str], ...]:
    phase1 = inspection.data.get("phase1_status")
    if isinstance(phase1, dict) and phase1.get("next_available_stage") == "none_phase1_complete":
        return (("Technical", "Phase 1 complete / Stage 10+ not built yet"), ("Operator", "Review Stage 09 keyframes"))
    if isinstance(phase1, dict) and phase1.get("paused_reason"):
        return (("Technical", "Restore image backend and rerun Stage 09"), ("Reason", str(phase1.get("paused_reason"))))
    return (("Technical", "Stage 09: image/keyframe jobs"), ("Operator", "Inspect manifest status"))


def _keyframe_artifact_line(inspection: RunInspection, scenes: tuple[WorkspaceSceneData, ...]) -> tuple[str, bool]:
    if scenes:
        ok = sum(1 for scene in scenes if scene.file_exists)
        total = len(scenes)
        mark = "✓" if ok == total and total else "○"
        return (f"{mark} {ok}/{total} keyframe files", ok == total and total > 0)
    keyframes = sum(1 for scene_id in ("scene_01", "scene_02", "scene_03") if inspection.artifacts.get(f"keyframes/{scene_id}.png"))
    return (f"{'✓' if keyframes == 3 else '○'} {keyframes} keyframes", keyframes == 3)


def _motion_items(inspection: RunInspection) -> list[WorkspaceSceneData]:
    manifest = inspection.data.get("keyframe_manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("jobs"), list):
        items: list[WorkspaceSceneData] = []
        for job in manifest["jobs"]:
            if not isinstance(job, dict):
                continue
            status = str(job.get("status") or "queued")
            keyframe = str(job.get("output_path") or "")
            file_exists = Path(keyframe).exists() if keyframe else False
            if status == "finished" and not file_exists:
                status = "error"
                previous_error = str(job.get("error") or "").strip()
                job_error = "finished job output missing" + (f" · {previous_error}" if previous_error else "")
            else:
                job_error = str(job.get("error") or "")
            if status != "finished" and not file_exists:
                keyframe = "missing"
            items.append(
                WorkspaceSceneData(
                    scene_id=str(job.get("scene_id") or "unknown"),
                    keyframe=keyframe,
                    summary=str(job.get("prompt") or job.get("error") or "image job present"),
                    state_label="KEYFRAME JOB",
                    status=status,
                    backend=str(job.get("backend") or manifest.get("backend") or "not_checked"),
                    backend_status=str(manifest.get("backend_status") or "not_checked"),
                    overall_status=str(manifest.get("overall_status") or "not_checked"),
                    progress_percent="" if job.get("progress_percent") in (None, "") else str(job.get("progress_percent")),
                    elapsed=str(job.get("elapsed") or ""),
                    output_path=str(job.get("output_path") or ""),
                    error=job_error,
                    backend_job_id=str(job.get("backend_job_id") or ""),
                    file_exists=file_exists,
                    file_size_bytes=int(job.get("file_size_bytes") or (Path(keyframe).stat().st_size if file_exists else 0)),
                    file_mtime=str(job.get("file_mtime") or ""),
                    gallery_path=str(manifest.get("gallery_path") or ""),
                )
            )
        return items
    if isinstance(inspection.data.get("phase1_status"), dict):
        return []
    prompts = inspection.data.get("ltx_motion_prompts") or []
    if not isinstance(prompts, list):
        prompt_items = inspection.data.get("zimage_prompts") or []
        if not isinstance(prompt_items, list):
            return []
        return [
            WorkspaceSceneData(
                scene_id=str(prompt.get("scene_id") or "unknown"),
                keyframe="missing",
                summary=str(prompt.get("prompt") or prompt.get("model_prompt") or "image prompt present"),
                state_label="IMAGE PROMPT",
                status=str(prompt.get("status") or "queued"),
            )
            for prompt in prompt_items
            if isinstance(prompt, dict)
        ]
    items: list[WorkspaceSceneData] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        items.append(
            WorkspaceSceneData(
                scene_id=str(prompt.get("scene_id") or "unknown"),
                keyframe=str(prompt.get("source_keyframe_path") or ""),
                summary=str(prompt.get("camera_motion") or prompt.get("motion_prompt") or "motion prompt present"),
            )
        )
    return items


def _artifact_mark(inspection: RunInspection, name: str) -> str:
    return "✓" if inspection.artifacts.get(name) else "○"


def _run_type(data_source_path: Path) -> str:
    if not data_source_path.exists():
        return "missing"
    if (data_source_path / "creative_os").is_dir():
        return "creative_os"
    if any((data_source_path / name).exists() for name in ("result.json", "plan.json", "state.json")):
        return "agent_core"
    return "unknown"


def _session_mode(data_source_path: Path, run_found: bool) -> str:
    if not run_found:
        return "missing"
    if FIXTURE_MARKER in str(data_source_path):
        return "fixture/demo"
    return "real_run"


def _session_label(session_mode: str) -> str:
    return "real run" if session_mode == "real_run" else session_mode


def _watch_label(watch_enabled: bool, refresh_sec: float) -> str:
    if not watch_enabled:
        return "off"
    return f"on / {refresh_sec:g}s"


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _agent_core_data(data_source_path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for name in (
        "result.json",
        "state.json",
        "plan.json",
        "scene_plan.json",
        "model_prompts.json",
        "prompt_audit.json",
        "checkpoints.json",
        "decision_log.json",
        "director_output.json",
        "stage_contracts.json",
    ):
        loaded = _read_json(data_source_path / name)
        data[name] = loaded if isinstance(loaded, dict) else {}
    return data


def _nested(data: dict[str, Any], *keys: str) -> object:
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _agent_core_resolution(plan: dict[str, Any]) -> str:
    width = plan.get("width")
    height = plan.get("height")
    if width and height:
        return f"{width}x{height}"
    return str(plan.get("resolution_label") or "unknown")


def _director_status(director_mode: str, director_llm_active: bool) -> str:
    if director_llm_active:
        return f"✓ {director_mode}"
    if director_mode == "rule_based_fallback":
        return "✗ rule_based_fallback"
    return f"? {director_mode}"


def _agent_core_artifacts(data_source_path: Path) -> tuple[tuple[str, bool], ...]:
    names = (
        "result.json",
        "state.json",
        "plan.json",
        "scene_plan.json",
        "model_prompts.json",
        "prompt_audit.json",
        "director_output.json",
        "final.mp4",
    )
    return tuple((f"{'✓' if (data_source_path / name).exists() else '○'} {name}", (data_source_path / name).exists()) for name in names)


def _agent_core_issues(director_mode: str, director_llm_active: bool, final_mp4: bool) -> tuple[str, ...]:
    issues: list[str] = []
    if director_mode == "rule_based_fallback" or not director_llm_active:
        issues.extend(
            (
                "DIRECTOR FALLBACK",
                f"director_llm_active={str(director_llm_active).lower()}",
                "check 127.0.0.1:8011 before production run",
            )
        )
    if not final_mp4:
        issues.append("final.mp4 missing")
    return tuple(issues)


def _agent_core_scenes(scene_plan: dict[str, Any], plan: dict[str, Any]) -> tuple[WorkspaceSceneData, ...]:
    scenes = scene_plan.get("scenes") or plan.get("scenes") or []
    if not isinstance(scenes, list):
        return tuple()
    items: list[WorkspaceSceneData] = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{index:02d}")
        summary = str(scene.get("description") or scene.get("prompt_text") or scene.get("title") or "scene plan present")
        items.append(
            WorkspaceSceneData(
                scene_id=scene_id,
                keyframe="agent-core scene plan",
                summary=summary,
                state_label="AGENT CORE PLAN",
                status="read-only scene plan",
            )
        )
    return tuple(items)


def _agent_core_pipeline_map(checkpoints: dict[str, Any], state: dict[str, Any]) -> tuple[PipelineStageData, ...]:
    raw = checkpoints.get("checkpoints") or state.get("checkpoints") or []
    if isinstance(raw, dict):
        raw_items = list(raw.values())
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []
    stages: list[PipelineStageData] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        stages.append(
            PipelineStageData(
                f"{index:02d}",
                str(item.get("checkpoint_id") or item.get("id") or item.get("stage") or "checkpoint"),
                str(item.get("status") or "unknown"),
            )
        )
    return tuple(stages)


def _agent_core_last_passed(checkpoints: dict[str, Any]) -> str:
    stages = _agent_core_pipeline_map(checkpoints, {})
    passed = [stage for stage in stages if stage.status == "passed"]
    if not passed:
        return "- none"
    stage = passed[-1]
    return f"✓ {stage.index} {stage.name}"


def _agent_core_next(issues: tuple[str, ...], final_mp4: bool) -> str:
    if "DIRECTOR FALLBACK" in issues:
        return "restore/check Director on 8011"
    if not final_mp4:
        return "run did not complete final output"
    return "inspect completed output"


def _agent_core_next_rows(issues: tuple[str, ...], final_mp4: bool) -> tuple[tuple[str, str], ...]:
    if "DIRECTOR FALLBACK" in issues:
        return (("Technical", "restore/check Director on 8011"), ("Run", "read-only inspect"))
    if not final_mp4:
        return (("Technical", "run did not complete final output"), ("Run", "read-only inspect"))
    return (("Technical", "inspect completed output"), ("Run", "read-only inspect"))


def _issues_severity(issues: tuple[str, ...]) -> str:
    if not issues:
        return "none"
    lowered = " ".join(issue.lower() for issue in issues)
    if any(marker in lowered for marker in ("run not found", "unknown run type", "hard failure", "failed", "blocking missing")):
        return "error"
    if any(marker in lowered for marker in ("director fallback", "director_llm_active=false", "final.mp4 missing", "fallback")):
        return "warning"
    return "error"
