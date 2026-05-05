from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.creative_system.skill_loader import SkillDocument, resolve_skills_for_pipeline


SKILL_CATEGORY_KEYS = {
    "platforms/": "platform_skills",
    "models/": "model_skills",
    "stages/": "stage_skills",
    "review/": "review_skills",
    "directing/": "directing_skills",
}


@dataclass(frozen=True)
class SkillInjectionContext:
    pipeline_id: str
    mode_id: str
    style_id: str
    required_skills: list[str] = field(default_factory=list)
    loaded_skills: list[dict[str, Any]] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    platform_skills: list[str] = field(default_factory=list)
    model_skills: list[str] = field(default_factory=list)
    stage_skills: list[str] = field(default_factory=list)
    review_skills: list[str] = field(default_factory=list)
    directing_skills: list[str] = field(default_factory=list)
    prompt_policy: dict[str, Any] = field(default_factory=dict)
    creative_constraints: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    audit_hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "mode_id": self.mode_id,
            "style_id": self.style_id,
            "required_skills": list(self.required_skills),
            "loaded_skills": list(self.loaded_skills),
            "missing_skills": list(self.missing_skills),
            "platform_skills": list(self.platform_skills),
            "model_skills": list(self.model_skills),
            "stage_skills": list(self.stage_skills),
            "review_skills": list(self.review_skills),
            "directing_skills": list(self.directing_skills),
            "prompt_policy": dict(self.prompt_policy),
            "creative_constraints": list(self.creative_constraints),
            "anti_patterns": list(self.anti_patterns),
            "audit_hints": list(self.audit_hints),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def build_skill_injection_context(
    *,
    pipeline_def: Any,
    mode: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    job_metadata: dict[str, Any] | None = None,
) -> SkillInjectionContext:
    mode = dict(mode or {})
    style = dict(style or {})
    metadata = dict(job_metadata or {})
    warnings: list[str] = []
    try:
        result = resolve_skills_for_pipeline(pipeline_def, mode, style)
    except Exception as exc:  # pragma: no cover - defensive fallback
        result = None
        warnings.append(f"skill_resolution_failed:{exc}")

    loaded_docs: list[SkillDocument] = list(result.loaded) if result is not None else []
    required_skills = list(result.required_skill_ids) if result is not None else [
        str(skill_id) for skill_id in getattr(pipeline_def, "required_skills", []) or []
    ]
    missing_skills = list(result.missing) if result is not None else []
    loaded_traces = [skill.to_trace() for skill in loaded_docs]
    loaded_ids = [skill.skill_id for skill in loaded_docs]
    categorized = {key: [] for key in SKILL_CATEGORY_KEYS.values()}
    for skill_id in loaded_ids:
        for prefix, key in SKILL_CATEGORY_KEYS.items():
            if skill_id.startswith(prefix):
                categorized[key].append(skill_id)

    prompt_policy = dict(mode.get("backend_prompt_policy") or {})
    if metadata.get("backend_prompt_policy") and isinstance(metadata.get("backend_prompt_policy"), dict):
        prompt_policy.update(metadata["backend_prompt_policy"])
    creative_constraints = _unique_strings(
        [
            *list(mode.get("global_forbidden") or []),
            *list(style.get("principles") or []),
            *list(style.get("human_visibility_rules") or []),
        ]
    )
    anti_patterns = _unique_strings([*list(mode.get("anti_patterns") or []), *list(metadata.get("anti_patterns") or [])])
    audit_hints = _unique_strings(
        [
            hint
            for skill in loaded_docs
            for hint in skill.audit_hints
        ]
    )
    if missing_skills:
        warnings.append("missing_optional_or_required_skills_recorded")

    return SkillInjectionContext(
        pipeline_id=str(getattr(pipeline_def, "pipeline_id", "") or metadata.get("pipeline_id") or ""),
        mode_id=str(mode.get("mode_id") or metadata.get("mode_id") or ""),
        style_id=str(style.get("style_id") or mode.get("visual_style") or metadata.get("style_id") or ""),
        required_skills=required_skills,
        loaded_skills=loaded_traces,
        missing_skills=missing_skills,
        platform_skills=categorized["platform_skills"],
        model_skills=categorized["model_skills"],
        stage_skills=categorized["stage_skills"],
        review_skills=categorized["review_skills"],
        directing_skills=categorized["directing_skills"],
        prompt_policy=prompt_policy,
        creative_constraints=creative_constraints,
        anti_patterns=anti_patterns,
        audit_hints=audit_hints,
        warnings=warnings,
        metadata={
            "mode_pacing": mode.get("pacing") or {},
            "motif_families": list(mode.get("motif_families") or []),
            "stage_roles": dict(getattr(pipeline_def, "stage_roles", {}) or {}),
            "source": "pipeline_def_plus_mode_style_job_metadata",
        },
    )


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out
