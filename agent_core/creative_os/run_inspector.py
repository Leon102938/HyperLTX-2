from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LEGACY_STAGES = [
    ("01", "Input normalized", "normalized_job.json"),
    ("02", "Intent routed", "intent_route.json"),
    ("03", "Skills loaded", "skill_match.json"),
    ("04", "Creative strategy", "creative_strategy.json"),
    ("05", "HiDream prompts", "hidream_prompts.json"),
    ("06", "Keyframes generated", "keyframe_manifest.json"),
    ("07", "Keyframe QA", "keyframe_review.json"),
    ("08", "LTX motion prompts", "ltx_motion_prompts.json"),
    ("09", "LTX video takes", "ltx_video_takes_manifest.json"),
    ("10", "Video review", "video_review.json"),
    ("11", "Assembly", "final.mp4"),
    ("12", "Final verdict", "final_quality_verdict.json"),
]

PHASE1_STAGES = [
    ("00", "Command Center", "normalized_job.json"),
    ("01", "Pipeline Overview", "pipeline_route.json"),
    ("02", "Mode & Style", "mode_style.json"),
    ("03", "Skills laden", "skill_tree.json"),
    ("04", "Creative Strategy", "creative_strategy.json"),
    ("05", "Beat / Hook Planner", "beat_hook_plan.json"),
    ("06", "Creative Judge", "creative_judge.json"),
    ("07", "Scene Contracts", "scene_contracts.json"),
    ("08", "Prompt Compiler", "prompt_payload_compiled.json"),
    ("09", "Image / Keyframe Generation", "keyframe_manifest.json"),
]

STAGES = LEGACY_STAGES


@dataclass
class StageStatus:
    index: str
    name: str
    status: str
    artifact: str
    detail: str = ""


@dataclass
class RunInspection:
    job_id: str
    run_dir: Path
    exists: bool
    status: str = "unknown"
    stages: list[StageStatus] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    artifacts: dict[str, bool] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


class CreativeOSRunInspector:
    def __init__(self, *, runs_root: Path | str = "/workspace/agent_runs") -> None:
        self.runs_root = Path(runs_root)

    def inspect(self, job_id: str) -> RunInspection:
        run_dir = self.runs_root / job_id / "creative_os"
        inspection = RunInspection(job_id=job_id, run_dir=run_dir, exists=run_dir.exists())
        if not inspection.exists:
            inspection.status = "run_not_found"
            inspection.blocking_issues.append("run_not_found")
            return inspection

        self._load_known_json(inspection)
        inspection.artifacts = self._artifact_map(run_dir)
        inspection.stages = self._stage_statuses(inspection)
        inspection.issues = self._collect_issues(inspection)
        inspection.blocking_issues = [issue for issue in inspection.issues if issue.startswith("blocking:")]
        inspection.status = self._derive_status(inspection)
        return inspection

    def _load_known_json(self, inspection: RunInspection) -> None:
        for path in sorted(inspection.run_dir.glob("*.json")):
            try:
                inspection.data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                inspection.data[path.stem] = {"_load_error": str(exc)}

    def _artifact_map(self, run_dir: Path) -> dict[str, bool]:
        names = [
            "normalized_job.json",
            "pipeline_route.json",
            "mode_style.json",
            "creative_direction.json",
            "skill_tree.json",
            "intent_route.json",
            "skill_match.json",
            "creative_strategy.json",
            "beat_hook_plan.json",
            "selected_beat_plan.json",
            "creative_judge.json",
            "scene_contracts.json",
            "keyframe_contracts.json",
            "prompt_payload_compiled.json",
            "hidream_prompts.json",
            "keyframe_manifest.json",
            "phase1_status.json",
            "live_status.json",
            "stage_events.jsonl",
            "keyframe_gallery.html",
            "keyframe_review.json",
            "stage6_review_decision.json",
            "ltx_motion_prompts.json",
            "ltx_prompt_audit.json",
            "ltx_video_takes_manifest.json",
            "video_review.json",
            "final_quality_verdict.json",
            "creative_os_report.md",
            "creative_os_stage6_report.md",
            "creative_os_stage7_report.md",
            "final.mp4",
        ]
        artifacts = {name: (run_dir / name).exists() for name in names}
        for scene_id in ("scene_01", "scene_02", "scene_03"):
            artifacts[f"keyframes/{scene_id}.png"] = (run_dir / "keyframes" / f"{scene_id}.png").exists()
        return artifacts

    def _stage_statuses(self, inspection: RunInspection) -> list[StageStatus]:
        statuses: list[StageStatus] = []
        stages = PHASE1_STAGES if isinstance(inspection.data.get("phase1_status"), dict) or isinstance(inspection.data.get("live_status"), dict) else LEGACY_STAGES
        for index, name, artifact in stages:
            status, detail = self._live_stage_status(inspection, index)
            if not status:
                status, detail = self._single_stage_status(inspection, artifact)
            statuses.append(StageStatus(index=index, name=name, status=status, artifact=artifact, detail=detail))
        return statuses

    def _live_stage_status(self, inspection: RunInspection, index: str) -> tuple[str, str] | tuple[None, None]:
        live = inspection.data.get("live_status")
        if not isinstance(live, dict):
            return None, None
        stages = live.get("stages")
        if not isinstance(stages, dict) or not isinstance(stages.get(index), dict):
            return None, None
        stage = stages[index]
        status = str(stage.get("status") or "unknown")
        if status == "done":
            return "passed", "live done"
        if status == "running":
            return "running", "live running"
        if status == "error":
            return "needs_review", str(stage.get("error") or "live error")
        if status == "missing":
            return "missing", str(stage.get("error") or "live missing")
        if status == "pending":
            return "pending", "live pending"
        return "unknown", f"live {status}"

    def _single_stage_status(self, inspection: RunInspection, artifact: str) -> tuple[str, str]:
        run_dir = inspection.run_dir
        if artifact == "hidream_prompts.json":
            prompts = inspection.data.get("hidream_prompts")
            if isinstance(prompts, list) and prompts:
                return "passed", f"{len(prompts)} prompts"
            return ("missing", "missing or empty") if not (run_dir / artifact).exists() else ("unknown", "empty")
        if artifact == "keyframe_manifest.json":
            manifest = inspection.data.get("keyframe_manifest") or {}
            if isinstance(manifest, dict) and isinstance(manifest.get("jobs"), list):
                jobs = manifest["jobs"]
                finished = [item for item in jobs if item.get("status") == "finished" and item.get("output_path") and Path(str(item["output_path"])).exists()]
                errors = [item for item in jobs if item.get("status") == "error"]
                missing_outputs = [item for item in jobs if item.get("status") == "finished" and not Path(str(item.get("output_path") or "")).exists()]
                if finished and len(finished) == len(jobs):
                    return "passed", f"{len(finished)}/{len(jobs)} jobs finished"
                if missing_outputs:
                    return "needs_review", f"{len(missing_outputs)}/{len(jobs)} finished jobs missing output"
                if errors:
                    backend_reason = str(manifest.get("backend_reason") or errors[0].get("error") or "image backend unavailable")
                    return "needs_review", f"{len(errors)}/{len(jobs)} jobs error; {backend_reason}"
                queued = [item for item in jobs if item.get("status") in {"queued", "running"}]
                if queued:
                    return "pending", f"{len(queued)}/{len(jobs)} jobs queued"
                return "unknown", "manifest jobs present"
            items = manifest.get("generated_keyframes") if isinstance(manifest, dict) else None
            if isinstance(items, list) and items:
                ok = [item for item in items if item.get("success") and item.get("image_path") and Path(item["image_path"]).exists()]
                return ("passed", f"{len(ok)}/{len(items)} pngs") if len(ok) == len(items) else ("needs_review", f"{len(ok)}/{len(items)} pngs")
            return ("missing", "missing") if not (run_dir / artifact).exists() else ("unknown", "no manifest items")
        if artifact == "keyframe_review.json":
            reviews = inspection.data.get("keyframe_review")
            if isinstance(reviews, list) and reviews:
                statuses = [str(item.get("status") or "unknown") for item in reviews]
                if all(status == "passed" for status in statuses):
                    return "passed", f"{len(reviews)} passed"
                if any(status == "rejected" for status in statuses):
                    return "rejected", ", ".join(statuses)
                return "needs_review", ", ".join(statuses)
            return ("missing", "missing") if not (run_dir / artifact).exists() else ("unknown", "empty")
        if artifact == "ltx_motion_prompts.json":
            prompts = inspection.data.get("ltx_motion_prompts")
            audit = inspection.data.get("ltx_prompt_audit") or {}
            if isinstance(prompts, list) and prompts and audit.get("overall_status") == "passed":
                return "passed", f"{len(prompts)} prompts; audit passed"
            if isinstance(prompts, list) and prompts:
                return "needs_review", f"{len(prompts)} prompts; audit {audit.get('overall_status', 'unknown')}"
            return ("missing", "missing") if not (run_dir / artifact).exists() else ("unknown", "empty")
        if artifact == "ltx_video_takes_manifest.json":
            manifest = inspection.data.get("ltx_video_takes_manifest")
            if isinstance(manifest, dict):
                return "unknown", "present; V1 dashboard does not evaluate videos yet"
            return "pending", "not built"
        if artifact in {"video_review.json", "final_quality_verdict.json", "final.mp4"}:
            return ("passed", "present") if (run_dir / artifact).exists() else ("pending", "not built")
        return ("passed", "present") if (run_dir / artifact).exists() else ("missing", "missing")

    def _collect_issues(self, inspection: RunInspection) -> list[str]:
        issues: list[str] = []
        reviews = inspection.data.get("keyframe_review")
        if isinstance(reviews, list):
            for item in reviews:
                status = str(item.get("status") or "unknown")
                if status in {"needs_review", "rejected"}:
                    prefix = "blocking:" if status == "rejected" else "review:"
                    issues.append(f"{prefix} keyframe {item.get('scene_id')}: {status} {', '.join(item.get('issues') or [])}")
        audit = inspection.data.get("ltx_prompt_audit")
        if isinstance(audit, dict):
            for item in audit.get("scene_results") or []:
                status = str(item.get("status") or "unknown")
                if status in {"needs_review", "rejected"}:
                    prefix = "blocking:" if status == "rejected" else "review:"
                    issues.append(f"{prefix} ltx prompt {item.get('scene_id')}: {status} {', '.join(item.get('issues') or [])}")
        return issues

    def _derive_status(self, inspection: RunInspection) -> str:
        live = inspection.data.get("live_status")
        if isinstance(live, dict):
            status = str(live.get("status") or "unknown")
            if status == "complete":
                return "phase1_live_complete"
            if status == "paused_missing_backend":
                return "phase1_paused_missing_image_backend"
            if status in {"running", "pending"}:
                return "phase1_live_running"
            if status in {"error", "missing"}:
                return f"phase1_live_{status}"
            if status:
                return status
        phase1 = inspection.data.get("phase1_status")
        if isinstance(phase1, dict) and phase1.get("status"):
            status = str(phase1.get("status"))
            if status == "paused_missing_backend":
                return "phase1_paused_missing_image_backend"
            if status == "finished":
                return "phase1_finished_stage09"
            return status
        stage7 = next((stage for stage in inspection.stages if stage.index == "07"), None)
        stage8 = next((stage for stage in inspection.stages if stage.index == "08"), None)
        stage9 = next((stage for stage in inspection.stages if stage.index == "09"), None)
        stage12 = next((stage for stage in inspection.stages if stage.index == "12"), None)
        if stage12 and stage12.status == "passed":
            return "completed"
        if stage7 and stage7.status in {"needs_review", "rejected"}:
            return "blocked_by_keyframe_review"
        if stage8 and stage8.status in {"needs_review", "rejected"}:
            return "blocked_by_ltx_prompt_audit"
        if inspection.blocking_issues:
            return "blocked"
        if stage8 and stage8.status == "passed" and stage9 and stage9.status == "pending":
            return "ready_for_ltx_i2v_takes"
        if any(stage.status == "passed" for stage in inspection.stages):
            return "in_progress"
        return "unknown"

    def next_action(self, inspection: RunInspection) -> str:
        if not inspection.exists:
            return "Fix job id or run path."
        if inspection.blocking_issues:
            return "Resolve blocking Creative OS issues before continuing."
        if inspection.status == "ready_for_ltx_i2v_takes":
            return "Stage 09: render 1 LTX I2V take per scene"
        if inspection.status == "phase1_paused_missing_image_backend":
            return "Start/restore HiDream-O1-Dev backend, then rerun Phase 1 image generation."
        if inspection.status == "phase1_finished_stage09":
            return "Phase 1 complete through Stage 09; Stage 10+ runtime is not built."
        for stage in inspection.stages:
            if stage.status in {"missing", "needs_review", "rejected", "unknown"}:
                return f"Complete or review Stage {stage.index}: {stage.name}"
        return "No next action detected."

    def last_passed_stage(self, inspection: RunInspection) -> str:
        phase1 = inspection.data.get("phase1_status")
        if isinstance(phase1, dict):
            stage_id = str(phase1.get("last_completed_stage") or "")
            stage_names = {index: name for index, name, _artifact in PHASE1_STAGES}
            if stage_id in stage_names:
                return f"{stage_id} {stage_names[stage_id]}"
        passed = [stage for stage in inspection.stages if stage.status == "passed"]
        if not passed:
            return "none"
        stage = passed[-1]
        return f"{stage.index} {stage.name}"

    def skill_health(self, inspection: RunInspection) -> dict[str, Any]:
        match = inspection.data.get("skill_match") or {}
        loaded = list(match.get("loaded_skill_ids") or [])
        fallbacks = list(match.get("fallback_skill_ids") or [])
        missing = list(match.get("missing_skill_ids") or [])
        reasons = match.get("reasons") or {}
        explicit_missing_optional = match.get("missing_optional")
        explicit_blocking_missing = match.get("blocking_missing")

        if isinstance(explicit_missing_optional, list) or isinstance(explicit_blocking_missing, list):
            missing_optional = [str(skill_id) for skill_id in (explicit_missing_optional or [])]
            blocking_missing = [str(skill_id) for skill_id in (explicit_blocking_missing or [])]
        else:
            fallback_covered: set[str] = set()
            for reason in reasons.values():
                marker = "fallback for missing "
                if isinstance(reason, str) and marker in reason:
                    fallback_covered.add(reason.split(marker, 1)[1].strip())

            missing_optional = [skill_id for skill_id in missing if skill_id in fallback_covered]
            blocking_missing = [skill_id for skill_id in missing if skill_id not in fallback_covered]
        if blocking_missing:
            health = "blocked"
            mark = "✗"
        elif missing_optional and not fallbacks:
            health = "degraded"
            mark = "!"
        else:
            health = "ok"
            mark = "✓"

        return {
            "loaded": loaded,
            "fallbacks": fallbacks,
            "missing_optional": missing_optional,
            "blocking_missing": blocking_missing,
            "status": health,
            "mark": mark,
        }
