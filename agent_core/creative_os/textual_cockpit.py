from __future__ import annotations

try:
    from agent_core.creative_os.cockpit.app import CockpitArgs, CreativeOSCockpitApp, ThemePreviewApp, run_cockpit, run_theme_preview
    from agent_core.creative_os.cockpit.panel_types import CockpitPanel
    from agent_core.creative_os.cockpit.panels.common import scene_card_text as _scene_card_text
except ImportError as exc:  # pragma: no cover - exercised by script import guard.
    raise RuntimeError(
        "Textual is not installed. Install dependencies with: python3 -m pip install 'textual>=0.89,<1.0'"
    ) from exc

__all__ = [
    "CockpitArgs",
    "CockpitPanel",
    "CreativeOSCockpitApp",
    "ThemePreviewApp",
    "_scene_card_text",
    "run_cockpit",
    "run_theme_preview",
]
