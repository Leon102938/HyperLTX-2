from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STAGES = [
    ("01", "Input normalized", "normalized_job.json"),
    ("02", "Intent routed", "intent_route.json"),
    ("03", "Skills loaded", "skill_match.json"),
    ("04", "Creative strategy", "creative_strategy.json"),
    ("05", "Z-Image prompts", "zimage_prompts.json"),
    ("06", "Keyframes generated", "keyframe_manifest.json"),
    ("07", "Keyframe QA", "keyframe_review.json"),
    ("08", "LTX motion prompts", "ltx_motion_prompts.json"),
    ("09", "LTX video takes", "ltx_video_takes_manifest.json"),
    ("10", "Video review", "video_review.json"),
    ("11", "Assembly", "final.mp4"),
    ("12", "Final verdict", "final_quality_verdict.json"),
]


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
            "intent_route.json",
            "skill_match.json",
            "creative_strategy.json",
            "selected_beat_plan.json",
            "scene_contracts.json",
            "keyframe_contracts.json",
            "zimage_prompts.json",
            "keyframe_manifest.json",
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
        for index, name, artifact in STAGES:
            status, detail = self._single_stage_status(inspection, artifact)
            statuses.append(StageStatus(index=index, name=name, status=status, artifact=artifact, detail=detail))
        return statuses

    def _single_stage_status(self, inspection: RunInspection, artifact: str) -> tuple[str, str]:
        run_dir = inspection.run_dir
        if artifact == "zimage_prompts.json":
            prompts = inspection.data.get("zimage_prompts")
            if isinstance(prompts, list) and prompts:
                return "passed", f"{len(prompts)} prompts"
            return ("missing", "missing or empty") if not (run_dir / artifact).exists() else ("unknown", "empty")
        if artifact == "keyframe_manifest.json":
            manifest = inspection.data.get("keyframe_manifest") or {}
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
        for stage in inspection.stages:
            if stage.status in {"missing", "needs_review", "rejected", "unknown"}:
                return f"Complete or review Stage {stage.index}: {stage.name}"
        return "No next action detected."

    def last_passed_stage(self, inspection: RunInspection) -> str:
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
