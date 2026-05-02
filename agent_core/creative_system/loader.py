from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CreativeSystem:
    modes: dict[str, dict[str, Any]]
    styles: dict[str, dict[str, Any]]
    libraries: dict[str, dict[str, Any]]
    prompts: dict[str, str]

    def mode(self, mode_id: str) -> dict[str, Any]:
        return dict(self.modes.get(mode_id) or {})

    def style(self, style_id: str) -> dict[str, Any]:
        return dict(self.styles.get(style_id) or {})


def _load_jsonish(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def load_creative_system() -> CreativeSystem:
    modes = {path.stem: _load_jsonish(path) for path in (BASE_DIR / "modes").glob("*.yaml")}
    styles = {path.stem: _load_jsonish(path) for path in (BASE_DIR / "styles").glob("*.yaml")}
    libraries = {path.stem: _load_jsonish(path) for path in (BASE_DIR / "libraries").glob("*.yaml")}
    prompts = {path.stem: path.read_text(encoding="utf-8").strip() for path in (BASE_DIR / "prompts").glob("*.md")}
    return CreativeSystem(modes=modes, styles=styles, libraries=libraries, prompts=prompts)


def detect_mode_id(idea: str, script: str, metadata: dict[str, Any] | None = None) -> str:
    explicit = str((metadata or {}).get("mode_id") or "").strip()
    if explicit:
        return explicit
    text = f"{idea} {script}".lower()
    morning_markers = (
        "morning reset",
        "morgen",
        "vorhang",
        "curtain",
        "wasserglas",
        "water glass",
        "fenster",
        "window",
        "fokus-start",
        "focus start",
    )
    if any(marker in text for marker in morning_markers):
        return "morning_reset"
    return "generic_clean_social"
