from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from agent_core.creative_os.live_status import LIVE_STAGE_ARTIFACTS, LiveStatusWriter


DEFAULT_RUNS_ROOT = Path("/workspace/agent_runs")
DEFAULT_IMAGE_BACKEND_URL = os.environ.get("AGENT_CORE_BASE_URL", "http://127.0.0.1:8000")


@dataclass(frozen=True)
class Phase1RunConfig:
    job_id: str
    topic: str
    pipeline: str = "shortform_storyboard_v1"
    mode: str = "visual_adventure"
    style: str = "cinematic_nature"
    orientation: str = "portrait"
    duration_sec: int = 9
    scene_count: int = 3
    runs_root: Path = DEFAULT_RUNS_ROOT
    image_backend_url: str = DEFAULT_IMAGE_BACKEND_URL
    image_backend: str = "zimage_http"
    attempt_images: bool = True
    stage_delay_seconds: float = 0.0


@dataclass(frozen=True)
class RetryKeyframeConfig:
    job_id: str
    runs_root: Path = DEFAULT_RUNS_ROOT
    image_backend_url: str = DEFAULT_IMAGE_BACKEND_URL
    image_backend: str = "zimage_http"
    scene_id: str | None = None
    force: bool = False
    dry_run: bool = False


def run_phase1(config: Phase1RunConfig) -> dict[str, Any]:
    started = _utc_now()
    run_dir = config.runs_root / config.job_id / "creative_os"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "keyframes").mkdir(exist_ok=True)

    normalized_job = _normalized_job(config, started)
    pipeline_route = _pipeline_route(config)
    mode_style = _mode_style(config)
    skill_match, skill_tree = _skill_artifacts(config)
    creative_strategy = _creative_strategy(config, normalized_job, mode_style, skill_match)
    beat_hook_plan = _beat_hook_plan(config, creative_strategy)
    creative_judge = _creative_judge(config, creative_strategy, beat_hook_plan)
    scene_contracts = _scene_contracts(config, creative_judge)
    prompt_payload, zimage_prompts = _prompt_payload(config, scene_contracts)

    _write_json(run_dir / "normalized_job.json", normalized_job)
    _write_json(run_dir / "pipeline_route.json", pipeline_route)
    _write_json(run_dir / "intent_route.json", _intent_route(config, pipeline_route))
    _write_json(run_dir / "mode_style.json", mode_style)
    _write_json(run_dir / "creative_direction.json", mode_style)
    _write_json(run_dir / "skill_match.json", skill_match)
    _write_json(run_dir / "skill_tree.json", skill_tree)
    _write_json(run_dir / "creative_strategy.json", creative_strategy)
    _write_json(run_dir / "beat_hook_plan.json", beat_hook_plan)
    _write_json(run_dir / "selected_beat_plan.json", beat_hook_plan["selected_beat_plan"])
    _write_json(run_dir / "creative_judge.json", creative_judge)
    _write_json(run_dir / "stage6_review_decision.json", _stage6_compat_decision(creative_judge))
    _write_json(run_dir / "scene_contracts.json", scene_contracts)
    _write_json(run_dir / "keyframe_contracts.json", scene_contracts)
    _write_json(run_dir / "prompt_payload_compiled.json", prompt_payload)
    _write_json(run_dir / "zimage_prompts.json", zimage_prompts)

    manifest = _run_image_jobs(config, run_dir, zimage_prompts)
    gallery_path = _write_keyframe_gallery(run_dir, manifest)
    if gallery_path:
        manifest["gallery_path"] = str(gallery_path)
    _write_json(run_dir / "keyframe_manifest.json", manifest)
    _write_json(run_dir / "phase1_status.json", _phase1_status(config, started, manifest))

    return {
        "job_id": config.job_id,
        "run_dir": str(run_dir),
        "creative_os_dir": str(run_dir),
        "status": "paused_missing_image_backend" if manifest["backend_status"] != "available" else manifest["overall_status"],
        "backend_status": manifest["backend_status"],
        "artifacts": {
            "00": "normalized_job.json",
            "01": "pipeline_route.json",
            "02": "mode_style.json",
            "03": "skill_match.json / skill_tree.json",
            "04": "creative_strategy.json",
            "05": "beat_hook_plan.json",
            "06": "creative_judge.json",
            "07": "scene_contracts.json",
            "08": "prompt_payload_compiled.json / zimage_prompts.json",
            "09": "keyframe_manifest.json",
        },
    }


def run_phase1_live(config: Phase1RunConfig) -> dict[str, Any]:
    started = _utc_now()
    run_dir = config.runs_root / config.job_id / "creative_os"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "keyframes").mkdir(exist_ok=True)
    live = LiveStatusWriter(run_dir=run_dir, job_id=config.job_id)
    live.initialize(viewed_stage="00")

    try:
        live.stage_running("00")
        normalized_job = _normalized_job(config, started)
        normalized_job["command_center"]["source"] = "agent_core_cli creative-os run-phase1-live"
        _write_live_stage(run_dir, live, "00", "normalized_job.json", normalized_job)
        _stage_delay(config)

        live.stage_running("01")
        pipeline_route = _pipeline_route(config)
        _write_json(run_dir / "pipeline_route.json", pipeline_route)
        _write_json(run_dir / "intent_route.json", _intent_route(config, pipeline_route))
        _finish_live_artifact(run_dir, live, "01", "pipeline_route.json")
        _stage_delay(config)

        live.stage_running("02")
        mode_style = _mode_style(config)
        _write_json(run_dir / "mode_style.json", mode_style)
        _write_json(run_dir / "creative_direction.json", mode_style)
        _finish_live_artifact(run_dir, live, "02", "mode_style.json")
        _stage_delay(config)

        live.stage_running("03")
        skill_match, skill_tree = _skill_artifacts(config)
        _write_json(run_dir / "skill_match.json", skill_match)
        _write_json(run_dir / "skill_tree.json", skill_tree)
        _finish_live_artifact(run_dir, live, "03", "skill_tree.json")
        _stage_delay(config)

        live.stage_running("04")
        creative_strategy = _creative_strategy(config, normalized_job, mode_style, skill_match)
        _write_live_stage(run_dir, live, "04", "creative_strategy.json", creative_strategy)
        _stage_delay(config)

        live.stage_running("05")
        beat_hook_plan = _beat_hook_plan(config, creative_strategy)
        _write_json(run_dir / "beat_hook_plan.json", beat_hook_plan)
        _write_json(run_dir / "selected_beat_plan.json", beat_hook_plan["selected_beat_plan"])
        _finish_live_artifact(run_dir, live, "05", "beat_hook_plan.json")
        _stage_delay(config)

        live.stage_running("06")
        creative_judge = _creative_judge(config, creative_strategy, beat_hook_plan)
        _write_json(run_dir / "creative_judge.json", creative_judge)
        _write_json(run_dir / "stage6_review_decision.json", _stage6_compat_decision(creative_judge))
        _finish_live_artifact(run_dir, live, "06", "creative_judge.json")
        _stage_delay(config)

        live.stage_running("07")
        scene_contracts = _scene_contracts(config, creative_judge)
        _write_json(run_dir / "scene_contracts.json", scene_contracts)
        _write_json(run_dir / "keyframe_contracts.json", scene_contracts)
        _finish_live_artifact(run_dir, live, "07", "scene_contracts.json")
        _stage_delay(config)

        live.stage_running("08")
        prompt_payload, zimage_prompts = _prompt_payload(config, scene_contracts)
        _write_json(run_dir / "prompt_payload_compiled.json", prompt_payload)
        _write_json(run_dir / "zimage_prompts.json", zimage_prompts)
        _finish_live_artifact(run_dir, live, "08", "prompt_payload_compiled.json")
        _stage_delay(config)

        live.stage_running("09")
        manifest = _run_image_jobs(config, run_dir, zimage_prompts, on_update=lambda payload: _write_json(run_dir / "keyframe_manifest.json", payload))
        gallery_path = _write_keyframe_gallery(run_dir, manifest)
        if gallery_path:
            manifest["gallery_path"] = str(gallery_path)
        _write_json(run_dir / "keyframe_manifest.json", manifest)
        _write_json(run_dir / "phase1_status.json", _phase1_status(config, started, manifest))
        if _stage09_manifest_complete(manifest):
            _finish_live_artifact(run_dir, live, "09", "keyframe_manifest.json")
        else:
            live.stage_error("09", artifact_path=run_dir / "keyframe_manifest.json", error=_stage09_live_error(manifest))
        live.finish(status="complete" if manifest["overall_status"] == "finished" else manifest["overall_status"])

        return _phase1_summary(config, run_dir, manifest)
    except Exception as exc:
        current = str(live.read().get("current_running_stage") or live.read().get("real_run_stage") or "00")
        live.stage_error(current, artifact_path=run_dir / LIVE_STAGE_ARTIFACTS.get(current, "unknown"), error=str(exc))
        live.finish(status="error", error=str(exc))
        raise


def retry_keyframes(config: RetryKeyframeConfig) -> dict[str, Any]:
    run_dir = config.runs_root / config.job_id / "creative_os"
    manifest_path = run_dir / "keyframe_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing keyframe_manifest.json for {config.job_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        raise RuntimeError("keyframe_manifest.json has no jobs list")

    run_config = _phase1_config_from_run(config, run_dir, manifest)
    prompts_by_scene = _prompts_by_scene(run_dir)
    retry_plan, skipped = _keyframe_retry_plan(run_dir, manifest, config)
    summary: dict[str, Any] = {
        "job_id": config.job_id,
        "run_dir": str(run_dir),
        "dry_run": config.dry_run,
        "force": config.force,
        "scene": config.scene_id,
        "retry_jobs": retry_plan,
        "skipped_jobs": skipped,
        "updated_files": [],
    }
    if config.dry_run:
        summary["status"] = "dry_run"
        return summary
    if not retry_plan:
        _refresh_manifest_files(run_dir, manifest)
        _write_json(manifest_path, manifest)
        _write_json(run_dir / "phase1_status.json", _phase1_status(run_config, _phase1_started_at(run_dir), manifest))
        summary["status"] = manifest.get("overall_status", "not_checked")
        summary["updated_files"] = ["keyframe_manifest.json", "phase1_status.json"]
        return summary

    started = time.monotonic()
    backend = _probe_zimage(run_config.image_backend_url)
    retry_scenes = {item["scene_id"] for item in retry_plan}
    for job in manifest["jobs"]:
        if not isinstance(job, dict) or str(job.get("scene_id") or "") not in retry_scenes:
            continue
        scene_id = str(job.get("scene_id") or "")
        output_path = Path(str(job.get("output_path") or run_dir / "keyframes" / f"{scene_id}.png"))
        prompt_text = str(job.get("prompt") or prompts_by_scene.get(scene_id) or "")
        if not prompt_text:
            job.update({"status": "error", "progress_percent": None, "elapsed": "00:00", "error": "missing prompt for retry"})
            job.update(_output_file_metadata(output_path))
            continue
        job.update(
            {
                "prompt": prompt_text,
                "backend": run_config.image_backend,
                "status": "queued",
                "output_path": str(output_path),
                "progress_percent": None,
                "elapsed": "00:00",
                "error": None,
                "backend_job_id": None,
            }
        )
        if backend["available"]:
            _submit_zimage_job(run_config, {"scene_id": scene_id, "prompt": prompt_text}, output_path, job, started)
        else:
            job["status"] = "error"
            job["error"] = backend["reason"]
        job.update(_output_file_metadata(output_path))

    manifest["backend"] = run_config.image_backend
    manifest["backend_url"] = run_config.image_backend_url
    manifest["backend_status"] = "available" if backend["available"] else "missing"
    manifest["backend_reason"] = backend["reason"]
    _refresh_manifest_files(run_dir, manifest)
    _write_json(manifest_path, manifest)
    _write_json(run_dir / "phase1_status.json", _phase1_status(run_config, _phase1_started_at(run_dir), manifest))
    summary["status"] = manifest["overall_status"]
    summary["backend_status"] = manifest["backend_status"]
    summary["updated_files"] = ["keyframe_manifest.json", "phase1_status.json"]
    if manifest.get("gallery_path"):
        summary["updated_files"].append("keyframe_gallery.html")
    return summary


def _normalized_job(config: Phase1RunConfig, started: str) -> dict[str, Any]:
    resolution = "512x768" if config.orientation == "portrait" else "768x512" if config.orientation == "landscape" else "768x768"
    return {
        "stage": "00",
        "job_id": config.job_id,
        "topic": config.topic,
        "pipeline": config.pipeline,
        "mode": config.mode,
        "style": config.style,
        "orientation": config.orientation,
        "format": config.orientation,
        "resolution": resolution,
        "duration_sec": config.duration_sec,
        "scene_count": config.scene_count,
        "output_targets": ["pipeline_route", "mode_style", "skills", "strategy", "beats", "judge", "scene_contracts", "image_prompts", "keyframes"],
        "created_at": started,
        "command_center": {"source": "agent_core_cli creative-os run-phase1", "execution": "cli"},
    }


def _pipeline_route(config: Phase1RunConfig) -> dict[str, Any]:
    flow = [
        ("00", "Command Center", "done", "job normalized"),
        ("01", "Pipeline Overview", "done", "selected pipeline route"),
        ("02", "Mode & Style", "done", "direction inputs"),
        ("03", "Skills laden", "done", "mode/style/hook/model skills"),
        ("04", "Creative Strategy", "done", "strategy pattern"),
        ("05", "Beat / Hook Planner", "done", "hook and beats"),
        ("06", "Creative Judge", "done", "decision and fixes"),
        ("07", "Scene Contracts", "done", "scene rules"),
        ("08", "Prompt Compiler", "done", "image prompts"),
        ("09", "Image / Keyframe Generation", "current", "image job manifest"),
        ("10-15", "Video / Final", "planned", "not built in Phase 1"),
    ]
    return {
        "stage": "01",
        "selected_pipeline": config.pipeline,
        "pipeline": config.pipeline,
        "status": "selected",
        "assets": {
            "keyframes": "image workflow",
            "voice": "optional",
            "music": "optional",
            "subtitles": "optional",
            "scene_count": config.scene_count,
            "final_mp4": "planned_after_phase1",
        },
        "flow": [{"stage": stage, "name": name, "status": status, "description": description} for stage, name, status, description in flow],
    }


def _intent_route(config: Phase1RunConfig, pipeline_route: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "01",
        "topic_intent": config.topic,
        "mode_intent": config.mode,
        "genre_intent": config.mode,
        "style_intent": config.style,
        "selected_pipeline": config.pipeline,
        "flow": pipeline_route["flow"],
    }


def _mode_style(config: Phase1RunConfig) -> dict[str, Any]:
    return {
        "stage": "02",
        "mode": config.mode,
        "style": config.style,
        "intent": f"Create a short, visual-first story about {config.topic}.",
        "visual_language": ["cinematic depth", "clear subject motion", "controlled camera", config.style],
        "risks_avoids": ["readable text", "UI screens", "paper labels", "muddy subject silhouette"],
        "handoff_to_skills": ["mode", "style", "hook", "model_prompting"],
        "status": "ready",
    }


def _skill_artifacts(config: Phase1RunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path("/workspace/agent_core/creative_system/skills")
    requested = {
        "mode_skills": [f"modes/{config.mode}", "stages/visual_direction"],
        "style_skills": [f"styles/{config.style}", "directing/clean_lifestyle_direction"],
        "hook_creative_skills": ["stages/creative_strategy", "stages/beat_planning", "directing/shortform_visual_director"],
        "model_skills": ["stages/model_prompting", "prompting/positive_image_prompting", "prompting/negative_prompt_policy"],
    }
    loaded: list[str] = []
    missing: list[str] = []
    fallbacks: list[str] = []
    reasons: dict[str, str] = {}
    for group, skill_ids in requested.items():
        for skill_id in skill_ids:
            if _skill_exists(root, skill_id):
                loaded.append(skill_id)
            elif "/" in skill_id and (skill_id.startswith("modes/") or skill_id.startswith("styles/")):
                missing.append(skill_id)
                fallback = "stages/visual_direction" if skill_id.startswith("modes/") else "directing/clean_lifestyle_direction"
                fallbacks.append(fallback)
                reasons[skill_id] = f"fallback for missing {skill_id}"
            else:
                missing.append(skill_id)
                reasons[skill_id] = "not found"
    fallbacks = sorted(set(fallbacks))
    loaded = sorted(set(loaded + fallbacks))
    missing = sorted(set(missing))
    skill_match = {
        "stage": "03",
        "status": "ok",
        "loaded_skill_ids": loaded,
        "fallback_skill_ids": fallbacks,
        "missing_skill_ids": missing,
        "missing_optional": [skill for skill in missing if skill in reasons and reasons[skill].startswith("fallback")],
        "blocking_missing": [skill for skill in missing if not reasons.get(skill, "").startswith("fallback")],
        "reasons": reasons,
        "groups": requested,
        "note": "Pipeline has no skills; skills are selected from mode, style, hook/creative, and model needs.",
    }
    return skill_match, {"stage": "03", "skill_tree_v1": requested, "match": skill_match}


def _creative_strategy(config: Phase1RunConfig, job: dict[str, Any], mode_style: dict[str, Any], skill_match: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "04",
        "status": "ready",
        "inputs": {"job_id": config.job_id, "pipeline": config.pipeline, "mode": config.mode, "style": config.style, "skills_loaded": len(skill_match["loaded_skill_ids"])},
        "strategy_pattern": "three-beat visual escalation",
        "core_idea": f"{config.topic} unfolds through a clear setup, discovery, and payoff.",
        "camera_visual_rules": ["one clear subject per scene", "no readable text", "strong foreground/midground/background", "motion cue per shot"],
        "output_readiness": "ready_for_beat_hook_planner",
    }


def _beat_hook_plan(config: Phase1RunConfig, strategy: dict[str, Any]) -> dict[str, Any]:
    options = [
        {"id": "hook_a", "text": f"Open on an immediate visual mystery in {config.topic}."},
        {"id": "hook_b", "text": f"Start calm, then reveal movement inside {config.topic}."},
        {"id": "hook_c", "text": f"Use a strong foreground reveal to pull viewers into {config.topic}."},
    ]
    beats = ["setup: establish the world", "discovery: introduce motion and tension", "payoff: reveal the memorable final image"]
    return {
        "stage": "05",
        "status": "selected",
        "hook_brief": {"goal": "fast visual clarity", "topic": config.topic},
        "hook_options": options,
        "beat_candidates": beats,
        "selected_beat_plan": {"hook": options[0]["text"], "beats": beats, "status": "ready", "handoff": "creative_judge"},
    }


def _creative_judge(config: Phase1RunConfig, strategy: dict[str, Any], beat_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "06",
        "status": "approved",
        "decision": "approved",
        "checks": {"visual_clarity": "passed", "text_risk": "needs_guardrails", "scene_progression": "passed"},
        "risks": ["avoid readable text and signage", "keep subject readable at small mobile size"],
        "fixes": ["add explicit no-text rule to every scene contract", "anchor each scene with one main subject"],
        "concept_notes": [strategy["core_idea"], beat_plan["selected_beat_plan"]["hook"]],
        "handoff": "scene_contracts",
    }


def _stage6_compat_decision(judge: dict[str, Any]) -> dict[str, Any]:
    return {"status": judge["status"], "decision": judge["decision"], "reviewer": "phase1_runtime", "issues": judge["risks"], "fixes": judge["fixes"]}


def _scene_contracts(config: Phase1RunConfig, judge: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = ["wide establishing reveal", "close subject action", "final cinematic payoff"]
    scenes: list[dict[str, Any]] = []
    for index in range(1, config.scene_count + 1):
        scene_id = f"scene_{index:02d}"
        anchor = anchors[(index - 1) % len(anchors)]
        scenes.append(
            {
                "stage": "07",
                "scene_id": scene_id,
                "title": f"{config.topic} / {anchor}",
                "visual_anchor": anchor,
                "environment": f"{config.topic}, natural environment, no signs or readable text",
                "action": ["establish location", "show subject motion", "resolve with memorable image"][(index - 1) % 3],
                "camera": ["slow push-in", "low tracking shot", "controlled reveal"][(index - 1) % 3],
                "lighting": "cinematic natural light",
                "allowed_visuals": ["natural subjects", "clear depth", "cinematic atmosphere"],
                "forbidden_visuals": ["readable text", "logos", "UI screens", "paper labels"],
                "text_glyph_risk": "blocked_by_prompt_rules",
                "status": "ready_for_prompt_compiler",
            }
        )
    return scenes


def _prompt_payload(config: Phase1RunConfig, scene_contracts: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    for scene in scene_contracts:
        prompt = (
            f"{scene['visual_anchor']} of {config.topic}, {scene['environment']}, "
            f"{scene['action']}, {scene['camera']}, {scene['lighting']}, {config.style}, "
            "high detail, clean subject, no readable text, no logos, no UI"
        )
        prompts.append(
            {
                "stage": "08",
                "scene_id": scene["scene_id"],
                "prompt": prompt,
                "model_prompt": prompt,
                "positive_prompt": prompt,
                "negative_prompt": "readable text, logos, watermarks, UI, paper labels, distorted letters",
                "backend": "zimage_http",
                "status": "compiled",
                "source_contract": "scene_contracts.json",
            }
        )
    return {
        "stage": "08",
        "status": "compiled",
        "image_prompts": prompts,
        "video_prompt_compiler": "pending_later",
        "audio_prompt_compiler": "pending_later",
        "music_prompt_compiler": "pending_later",
    }, prompts


def _run_image_jobs(
    config: Phase1RunConfig,
    run_dir: Path,
    prompts: list[dict[str, Any]],
    *,
    on_update: Any | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    backend = _probe_zimage(config.image_backend_url) if config.attempt_images else {"available": False, "reason": "disabled_by_cli"}
    jobs: list[dict[str, Any]] = []

    def refresh_manifest() -> dict[str, Any]:
        for item in jobs:
            item.update(_output_file_metadata(item.get("output_path")))
        manifest_payload = _keyframe_manifest(config, backend, jobs)
        if on_update is not None:
            on_update(manifest_payload)
        return manifest_payload

    for prompt in prompts:
        scene_id = str(prompt["scene_id"])
        output_path = run_dir / "keyframes" / f"{scene_id}.png"
        job = {
            "scene_id": scene_id,
            "prompt": prompt["prompt"],
            "backend": config.image_backend,
            "status": "queued",
            "output_path": str(output_path),
            "progress_percent": None,
            "elapsed": "00:00",
            "error": None,
            "backend_job_id": None,
        }
        jobs.append(job)
        refresh_manifest()
        if backend["available"]:
            _submit_zimage_job(config, prompt, output_path, job, started, on_update=refresh_manifest)
        else:
            job["status"] = "error"
            job["error"] = backend["reason"]
            refresh_manifest()

    return refresh_manifest()


def _keyframe_manifest(config: Phase1RunConfig, backend: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _manifest_overall_status(jobs, backend_available=bool(backend["available"]))
    return {
        "stage": "09",
        "backend": config.image_backend,
        "backend_url": config.image_backend_url,
        "backend_status": "available" if backend["available"] else "missing",
        "backend_reason": backend["reason"],
        "overall_status": overall,
        "jobs": jobs,
        "generated_keyframes": [{"scene_id": job["scene_id"], "success": job["status"] == "finished" and bool(job.get("file_exists")), "image_path": job["output_path"], "error": job["error"]} for job in jobs],
    }


def _submit_zimage_job(
    config: Phase1RunConfig,
    prompt: dict[str, Any],
    output_path: Path,
    job: dict[str, Any],
    started: float,
    *,
    on_update: Any | None = None,
) -> None:
    # Phase 1 supports the existing local zimage HTTP contract when present.
    payload = {
        "job_id": f"{config.job_id}_{prompt['scene_id']}",
        "prompt": prompt["prompt"],
        "width": 512 if config.orientation == "portrait" else 768,
        "height": 768 if config.orientation == "portrait" else 512,
        "steps": 9,
        "guidance_scale": 0.0,
    }
    try:
        submit = _http_json(f"{config.image_backend_url.rstrip('/')}/zimage/jobs", method="POST", payload=payload, timeout=60)
        backend_job_id = str(submit.get("job_id") or payload["job_id"])
        job["backend_job_id"] = backend_job_id
        if on_update is not None:
            on_update()
        latest: dict[str, Any] = {}
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            latest = _http_json(f"{config.image_backend_url.rstrip('/')}/zimage/jobs/{backend_job_id}", timeout=60)
            state = str(latest.get("state") or latest.get("status") or "")
            if state in {"succeeded", "failed", "error"}:
                break
            job["status"] = "running"
            progress = latest.get("progress_percent")
            if progress is None:
                progress = latest.get("progress")
            if progress is not None:
                job["progress_percent"] = progress
            if on_update is not None:
                on_update()
            time.sleep(2)
        if str(latest.get("state") or latest.get("status")) != "succeeded":
            job["status"] = "error"
            job["error"] = str(latest.get("error") or "zimage job failed")
            if on_update is not None:
                on_update()
            return
        source = Path(str(latest.get("output_path") or ""))
        if source.exists():
            if source.resolve() != output_path.resolve():
                output_path.write_bytes(source.read_bytes())
            job["status"] = "finished"
            job["progress_percent"] = 100
            job["elapsed"] = _format_elapsed(time.monotonic() - started)
            job["backend_job_id"] = backend_job_id
            if on_update is not None:
                on_update()
        else:
            job["status"] = "error"
            job["error"] = f"backend succeeded but output missing: {source}"
            if on_update is not None:
                on_update()
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        if on_update is not None:
            on_update()


def _probe_zimage(base_url: str) -> dict[str, Any]:
    try:
        payload = _http_json(f"{base_url.rstrip('/')}/DW/zimage_ready", timeout=5)
        if payload.get("ready"):
            return {"available": True, "reason": "ready"}
        return {"available": False, "reason": f"zimage readiness false: {payload}"}
    except Exception as exc:
        return {"available": False, "reason": f"zimage readiness probe failed: {exc}"}


def _phase1_status(config: Phase1RunConfig, started: str, manifest: dict[str, Any]) -> dict[str, Any]:
    finished = manifest["overall_status"] == "finished"
    completed = [f"{index:02d}" for index in range(0, 10 if finished else 9)]
    return {
        "job_id": config.job_id,
        "started_at": started,
        "updated_at": _utc_now(),
        "phase": "phase1_stage00_09",
        "status": manifest["overall_status"],
        "current_stage": "09",
        "real_run_stage": "09",
        "last_completed_stage": "09" if finished else "08",
        "next_available_stage": "none_phase1_complete" if finished else "09",
        "completed_stages": completed,
        "paused_reason": None if manifest["backend_status"] == "available" else manifest["backend_reason"],
        "stage09_manifest": "keyframe_manifest.json",
        "stage10_plus": "not_built",
    }


def _phase1_config_from_run(config: RetryKeyframeConfig, run_dir: Path, manifest: dict[str, Any]) -> Phase1RunConfig:
    job = _read_json(run_dir / "normalized_job.json")
    return Phase1RunConfig(
        job_id=config.job_id,
        topic=str(job.get("topic") or "unknown"),
        pipeline=str(job.get("pipeline") or "shortform_storyboard_v1"),
        mode=str(job.get("mode") or "visual_adventure"),
        style=str(job.get("style") or "cinematic_nature"),
        orientation=str(job.get("orientation") or job.get("format") or "portrait"),
        duration_sec=int(job.get("duration_sec") or 9),
        scene_count=int(job.get("scene_count") or len(manifest.get("jobs") or []) or 3),
        runs_root=config.runs_root,
        image_backend_url=config.image_backend_url,
        image_backend=str(manifest.get("backend") or config.image_backend),
        attempt_images=True,
    )


def _prompts_by_scene(run_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    payload = _read_json(run_dir / "zimage_prompts.json")
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("scene_id"):
                prompts[str(item["scene_id"])] = str(item.get("prompt") or item.get("model_prompt") or "")
    compiled = _read_json(run_dir / "prompt_payload_compiled.json")
    compiled_items = compiled.get("image_prompts") or [] if isinstance(compiled, dict) else []
    for item in compiled_items:
        if isinstance(item, dict) and item.get("scene_id") and str(item.get("prompt") or item.get("model_prompt") or ""):
            prompts[str(item["scene_id"])] = str(item.get("prompt") or item.get("model_prompt"))
    return prompts


def _keyframe_retry_plan(run_dir: Path, manifest: dict[str, Any], config: RetryKeyframeConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retry: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for job in manifest.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        scene_id = str(job.get("scene_id") or "")
        if config.scene_id and scene_id != config.scene_id:
            skipped.append({"scene_id": scene_id, "reason": "scene_filter"})
            continue
        output_path = str(job.get("output_path") or "")
        file_exists = Path(output_path).exists() if output_path else False
        status = str(job.get("status") or "queued")
        reason = _retry_reason(status, output_path, file_exists, config.force)
        item = {"scene_id": scene_id, "status": status, "output_path": output_path or "missing", "file_exists": file_exists, "reason": reason}
        if reason:
            retry.append(item)
        else:
            skipped.append({**item, "reason": "finished_file_exists"})
    return retry, skipped


def _retry_reason(status: str, output_path: str, file_exists: bool, force: bool) -> str:
    if force:
        return "force"
    if not output_path:
        return "missing_output_path"
    if status in {"failed", "error", "queued", "running"}:
        return status
    if not file_exists:
        return "missing_output_file"
    return ""


def _refresh_manifest_files(run_dir: Path, manifest: dict[str, Any]) -> None:
    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, dict)]
    for job in jobs:
        scene_id = str(job.get("scene_id") or "")
        if not job.get("output_path") and scene_id:
            job["output_path"] = str(run_dir / "keyframes" / f"{scene_id}.png")
        job.update(_output_file_metadata(job.get("output_path")))
        if job.get("status") == "finished" and not job.get("file_exists"):
            job["error"] = str(job.get("error") or "finished job output missing")
    backend_available = manifest.get("backend_status") == "available"
    manifest["overall_status"] = _manifest_overall_status(jobs, backend_available=backend_available)
    manifest["generated_keyframes"] = [
        {"scene_id": job.get("scene_id"), "success": job.get("status") == "finished" and bool(job.get("file_exists")), "image_path": job.get("output_path"), "error": job.get("error")}
        for job in jobs
    ]
    gallery_path = _write_keyframe_gallery(run_dir, manifest)
    if gallery_path:
        manifest["gallery_path"] = str(gallery_path)
    else:
        manifest.pop("gallery_path", None)


def _manifest_overall_status(jobs: list[dict[str, Any]], *, backend_available: bool) -> str:
    if jobs and all(job.get("status") == "finished" and bool(job.get("file_exists")) for job in jobs):
        return "finished"
    if not backend_available:
        return "paused_missing_backend"
    if any(job.get("status") in {"error", "failed"} for job in jobs):
        return "needs_review"
    if any(job.get("status") == "running" for job in jobs):
        return "running"
    if any(job.get("status") == "queued" for job in jobs):
        return "queued"
    if any(job.get("status") == "finished" and not bool(job.get("file_exists")) for job in jobs):
        return "needs_review"
    return "needs_review"


def _stage_delay(config: Phase1RunConfig) -> None:
    if config.stage_delay_seconds > 0:
        time.sleep(config.stage_delay_seconds)


def _stage09_manifest_complete(manifest: dict[str, Any]) -> bool:
    jobs = manifest.get("jobs")
    return (
        manifest.get("overall_status") == "finished"
        and isinstance(jobs, list)
        and bool(jobs)
        and all(isinstance(job, dict) and job.get("status") == "finished" and bool(job.get("file_exists")) for job in jobs)
    )


def _stage09_live_error(manifest: dict[str, Any]) -> str:
    backend_status = str(manifest.get("backend_status") or "unknown")
    backend_reason = str(manifest.get("backend_reason") or "").strip()
    jobs = [job for job in manifest.get("jobs") or [] if isinstance(job, dict)]
    if backend_status in {"missing", "disabled"} or backend_reason:
        return backend_reason or f"image backend {backend_status}"
    failed = [job for job in jobs if job.get("status") in {"error", "failed"}]
    if failed:
        return str(failed[0].get("error") or "keyframe job failed")
    missing_outputs = [job for job in jobs if not job.get("file_exists")]
    if missing_outputs:
        return "keyframe output missing"
    return "keyframe manifest not complete"


def _write_live_stage(run_dir: Path, live: LiveStatusWriter, stage_id: str, artifact: str, payload: Any) -> None:
    _write_json(run_dir / artifact, payload)
    _finish_live_artifact(run_dir, live, stage_id, artifact)


def _finish_live_artifact(run_dir: Path, live: LiveStatusWriter, stage_id: str, artifact: str) -> None:
    artifact_path = run_dir / artifact
    if artifact_path.exists():
        live.stage_done(stage_id, artifact_path=artifact_path)
    else:
        live.stage_missing(stage_id, artifact_path=artifact_path, error=f"missing artifact: {artifact}")


def _phase1_summary(config: Phase1RunConfig, run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": config.job_id,
        "run_dir": str(run_dir),
        "creative_os_dir": str(run_dir),
        "status": "paused_missing_image_backend" if manifest["backend_status"] != "available" else manifest["overall_status"],
        "backend_status": manifest["backend_status"],
        "artifacts": {
            "00": "normalized_job.json",
            "01": "pipeline_route.json",
            "02": "mode_style.json",
            "03": "skill_match.json / skill_tree.json",
            "04": "creative_strategy.json",
            "05": "beat_hook_plan.json",
            "06": "creative_judge.json",
            "07": "scene_contracts.json",
            "08": "prompt_payload_compiled.json / zimage_prompts.json",
            "09": "keyframe_manifest.json",
        },
    }


def _skill_exists(root: Path, skill_id: str) -> bool:
    return (root / f"{skill_id}.md").exists() or (root / skill_id).exists()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _phase1_started_at(run_dir: Path) -> str:
    status = _read_json(run_dir / "phase1_status.json")
    if isinstance(status, dict) and status.get("started_at"):
        return str(status["started_at"])
    job = _read_json(run_dir / "normalized_job.json")
    if isinstance(job, dict) and job.get("created_at"):
        return str(job["created_at"])
    return _utc_now()


def _write_keyframe_gallery(run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, dict)]
    image_jobs = [job for job in jobs if job.get("file_exists")]
    if not image_jobs:
        return None
    gallery = run_dir / "keyframe_gallery.html"
    rows = []
    for job in image_jobs:
        output_path = Path(str(job.get("output_path") or ""))
        try:
            rel = output_path.relative_to(run_dir)
        except ValueError:
            rel = output_path
        rows.append(
            "<figure>"
            f"<img src=\"{rel.as_posix()}\" alt=\"{job.get('scene_id')}\" style=\"max-width:240px\">"
            f"<figcaption>{job.get('scene_id')} · {job.get('status')} · {job.get('file_size_bytes')} bytes</figcaption>"
            "</figure>"
        )
    gallery.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Phase 1 Keyframes</title></head>"
        "<body><h1>Phase 1 Keyframes</h1><div style=\"display:flex;gap:16px;flex-wrap:wrap\">"
        + "".join(rows)
        + "</div></body></html>\n",
        encoding="utf-8",
    )
    return gallery


def _output_file_metadata(path_value: object) -> dict[str, Any]:
    if not path_value:
        return {"file_exists": False, "file_size_bytes": 0, "file_mtime": None}
    path = Path(str(path_value))
    exists = path.exists()
    if not exists:
        return {"file_exists": False, "file_size_bytes": 0, "file_mtime": None}
    stat = path.stat()
    return {
        "file_exists": True,
        "file_size_bytes": stat.st_size,
        "file_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    return json.loads(raw) if raw else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    return f"{minutes:02d}:{sec:02d}"
