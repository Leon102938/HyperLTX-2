from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SKILLS_DIR = Path(__file__).resolve().parent / "skills"
REQUIRED_FIELDS = (
    "title",
    "purpose",
    "when_to_use",
    "rules",
    "do",
    "dont",
    "output_contract",
    "common_failures",
    "audit_hints",
)


@dataclass(frozen=True)
class SkillDocument:
    skill_id: str
    path: str
    title: str
    purpose: str
    when_to_use: str
    rules: list[str]
    do: list[str]
    dont: list[str]
    output_contract: list[str]
    common_failures: list[str]
    audit_hints: list[str]
    raw_markdown: str
    missing_fields: list[str]

    def to_trace(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "path": self.path,
            "title": self.title,
            "purpose": self.purpose,
            "missing_fields": list(self.missing_fields),
        }


@dataclass(frozen=True)
class SkillLoadResult:
    required_skill_ids: list[str]
    loaded: list[SkillDocument]
    missing: list[str]

    def to_trace(self) -> dict[str, Any]:
        return {
            "required_skills": list(self.required_skill_ids),
            "loaded_skills": [skill.to_trace() for skill in self.loaded],
            "missing_skills": list(self.missing),
        }


def _normalize_skill_id(path_or_id: str | Path) -> str:
    value = str(path_or_id).strip()
    path = Path(value)
    if path.suffix == ".md":
        try:
            return str(path.resolve().relative_to(SKILLS_DIR.resolve())).removesuffix(".md")
        except ValueError:
            return path.stem
    return value.removesuffix(".md").strip("/")


def _resolve_skill_path(path_or_id: str | Path) -> tuple[str, Path]:
    skill_id = _normalize_skill_id(path_or_id)
    path = Path(path_or_id)
    if path.suffix == ".md" and path.is_absolute():
        return skill_id, path
    if path.suffix == ".md" and path.exists():
        return skill_id, path
    return skill_id, SKILLS_DIR / f"{skill_id}.md"


def _parse_markdown_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip().lower().replace(" ", "_")
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)
    return sections


def _section_text(sections: dict[str, list[str]], key: str) -> str:
    return "\n".join(line for line in sections.get(key, []) if line.strip()).strip()


def _section_list(sections: dict[str, list[str]], key: str) -> list[str]:
    items: list[str] = []
    for line in sections.get(key, []):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
        else:
            items.append(stripped)
    return items


def load_skill(path_or_id: str | Path) -> SkillDocument | None:
    skill_id, path = _resolve_skill_path(path_or_id)
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    sections = _parse_markdown_sections(raw)
    missing = [field for field in REQUIRED_FIELDS if not _section_text(sections, field)]
    return SkillDocument(
        skill_id=skill_id,
        path=str(path),
        title=_section_text(sections, "title") or skill_id,
        purpose=_section_text(sections, "purpose"),
        when_to_use=_section_text(sections, "when_to_use"),
        rules=_section_list(sections, "rules"),
        do=_section_list(sections, "do"),
        dont=_section_list(sections, "dont"),
        output_contract=_section_list(sections, "output_contract"),
        common_failures=_section_list(sections, "common_failures"),
        audit_hints=_section_list(sections, "audit_hints"),
        raw_markdown=raw,
        missing_fields=missing,
    )


def load_required_skills(skill_ids: list[str]) -> SkillLoadResult:
    required: list[str] = []
    loaded: list[SkillDocument] = []
    missing: list[str] = []
    for skill_id in skill_ids:
        normalized = _normalize_skill_id(skill_id)
        if normalized in required:
            continue
        required.append(normalized)
        skill = load_skill(normalized)
        if skill is None:
            missing.append(normalized)
        else:
            loaded.append(skill)
    return SkillLoadResult(required_skill_ids=required, loaded=loaded, missing=missing)


def resolve_skills_for_pipeline(pipeline_def: Any, mode: dict[str, Any] | str | None = None, style: dict[str, Any] | str | None = None) -> SkillLoadResult:
    skill_ids: list[str] = []
    for skill_id in getattr(pipeline_def, "required_skills", []) or []:
        skill_ids.append(str(skill_id))
    for step in getattr(pipeline_def, "steps", []) or []:
        for skill_id in getattr(step, "required_skills", []) or []:
            skill_ids.append(str(skill_id))

    if isinstance(mode, dict):
        for skill_id in mode.get("required_skills", []) or []:
            skill_ids.append(str(skill_id))
    if isinstance(style, dict):
        for skill_id in style.get("required_skills", []) or []:
            skill_ids.append(str(skill_id))

    return load_required_skills(skill_ids)
