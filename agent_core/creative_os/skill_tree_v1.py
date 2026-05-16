from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SKILL_ROOT = Path("/workspace/skills")
MANIFEST_NAME = "skill_manifest.json"


def load_skill_manifest(skill_root: Path = DEFAULT_SKILL_ROOT) -> dict[str, Any]:
    manifest_path = skill_root / MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "version": "missing",
            "status": "missing",
            "manifest_path": str(manifest_path),
            "categories": {},
            "error": f"missing manifest: {MANIFEST_NAME}",
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "version": "unknown",
            "status": "missing",
            "manifest_path": str(manifest_path),
            "categories": {},
            "error": f"manifest read error: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "version": "unknown",
            "status": "missing",
            "manifest_path": str(manifest_path),
            "categories": {},
            "error": "manifest root is not an object",
        }
    payload.setdefault("version", "unknown")
    payload.setdefault("categories", {})
    payload["status"] = "loaded"
    payload["manifest_path"] = str(manifest_path)
    return payload


def select_skill_ids(config: Any, manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    mode = str(getattr(config, "mode", "") or "unknown")
    style = str(getattr(config, "style", "") or "unknown")
    topic = str(getattr(config, "topic", "") or "").lower()
    hook = _select_hook(mode=mode, topic=topic)
    return {
        "modes": [{"id": mode, "reason": f"selected from job mode {mode}"}],
        "styles": [{"id": style, "reason": f"selected from job style {style}"}],
        "hooks": [{"id": hook, "reason": _hook_reason(hook, mode, topic)}],
        "models": [
            {"id": "hidream_o1_dev_prompt_rules", "reason": "Stage 08 compiles HiDream-O1-Dev image prompts"},
            {"id": "hidream_o1_storyboard_rules", "reason": "Stage 08 keeps image prompts tied to storyboard scene contracts"},
            {"id": "hidream_o1_no_unwanted_text_rules", "reason": "Stage 08 blocks text, poster, and typography artifacts for HiDream-O1-Dev"},
            {"id": "ltx_motion_rules", "reason": "Stage 07 prepares scene contracts for later motion handoff"},
            {"id": "qwen_tts_delivery_rules", "reason": "Phase 1 records audio/voice delivery rules as model context"},
        ],
    }


def load_skill_tree(config: Any, skill_root: Path = DEFAULT_SKILL_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_skill_manifest(skill_root)
    selections = select_skill_ids(config, manifest)
    categories = manifest.get("categories") if isinstance(manifest.get("categories"), dict) else {}
    loaded_skill_ids: list[str] = []
    missing_skill_ids: list[str] = []
    loaded_skills: dict[str, list[dict[str, Any]]] = {key: [] for key in ("modes", "styles", "hooks", "models")}
    missing_skills: list[dict[str, str]] = []
    reasons: dict[str, str] = {}

    for category, selected in selections.items():
        manifest_category = categories.get(category) if isinstance(categories.get(category), dict) else {}
        for item in selected:
            skill_id = str(item["id"])
            reason = str(item["reason"])
            manifest_path = manifest_category.get(skill_id)
            skill_ref = f"{category}/{skill_id}"
            if not manifest_path:
                missing_skill_ids.append(skill_ref)
                missing_skills.append({"category": category, "id": skill_id, "status": "missing", "reason": "not listed in manifest"})
                reasons[skill_ref] = "missing: not listed in manifest"
                continue
            path = skill_root / str(manifest_path)
            if not path.exists():
                missing_skill_ids.append(skill_ref)
                missing_skills.append({"category": category, "id": skill_id, "status": "missing", "reason": f"missing file: {manifest_path}"})
                reasons[skill_ref] = f"missing file: {manifest_path}"
                continue
            content = path.read_text(encoding="utf-8")
            rules = _extract_rules(content)
            loaded_skill_ids.append(skill_ref)
            reasons[skill_ref] = reason
            loaded_skills[category].append(
                {
                    "category": category,
                    "id": skill_id,
                    "skill_id": skill_ref,
                    "status": "loaded",
                    "path": str(path),
                    "reason": reason,
                    "rules": rules,
                    "content": content,
                }
            )

    status = "ok" if not missing_skill_ids and manifest.get("status") == "loaded" else "missing"
    skill_match = {
        "stage": "03",
        "version": "skill_tree_v1",
        "status": status,
        "manifest_status": manifest.get("status", "unknown"),
        "loaded_skill_ids": sorted(loaded_skill_ids),
        "fallback_skill_ids": [],
        "missing_skill_ids": sorted(missing_skill_ids),
        "missing_optional": sorted(missing_skill_ids),
        "blocking_missing": [],
        "reasons": reasons,
        "groups": {category: [entry["id"] for entry in entries] for category, entries in selections.items()},
        "selected": selections,
        "note": "Pipeline route is fixed; Skill Tree V1 only selects mode, style, hook, and model skills.",
    }
    skill_tree = {
        "stage": "03",
        "version": "skill_tree_v1",
        "status": status,
        "source": "skills/skill_manifest.json",
        "manifest": manifest,
        "selected": selections,
        "loaded_skills": loaded_skills,
        "missing_skills": missing_skills,
        "match": skill_match,
    }
    return skill_match, skill_tree


def skill_rules(skill_tree: dict[str, Any], categories: tuple[str, ...]) -> list[str]:
    loaded = skill_tree.get("loaded_skills")
    if not isinstance(loaded, dict):
        return []
    rules: list[str] = []
    for category in categories:
        for skill in loaded.get(category, []) or []:
            if isinstance(skill, dict):
                for rule in skill.get("rules", []) or []:
                    rules.append(str(rule))
    return rules


def skill_source_summary(skill_tree: dict[str, Any]) -> dict[str, Any]:
    loaded = skill_tree.get("loaded_skills") if isinstance(skill_tree.get("loaded_skills"), dict) else {}
    return {
        "source": "skills loaded",
        "version": str(skill_tree.get("version") or "unknown"),
        "loaded": {category: [str(item.get("skill_id")) for item in items if isinstance(item, dict)] for category, items in loaded.items()},
        "missing": skill_tree.get("missing_skills", []),
    }


def _select_hook(*, mode: str, topic: str) -> str:
    if mode == "finance_short" or "market" in topic or "finance" in topic:
        return "fast_market_hook"
    if mode == "calm_evergreen" or "calm" in topic or "morning routine" in topic:
        return "soft_observation_hook"
    if mode in {"practical_tip", "product_explainer"} or "tip" in topic or "benefit" in topic:
        return "benefit_hook"
    if "problem" in topic or "fix" in topic:
        return "small_problem_hook"
    return "curiosity_hook"


def _hook_reason(hook: str, mode: str, topic: str) -> str:
    if hook == "soft_observation_hook":
        return f"calm evergreen signal from mode/topic: {mode}"
    if hook == "fast_market_hook":
        return "finance or market signal in mode/topic"
    if hook == "benefit_hook":
        return "practical benefit signal in mode/topic"
    if hook == "small_problem_hook":
        return "small problem signal in topic"
    return "default shortform curiosity hook"


def _extract_rules(content: str) -> list[str]:
    rules: list[str] = []
    for line in content.splitlines():
        text = line.strip()
        if text.startswith("- "):
            rules.append(text[2:].strip())
    return rules
