from __future__ import annotations

from collections.abc import Iterator

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static

from agent_core.creative_os.cockpit.panel_registry import PANEL_REGISTRY
from agent_core.creative_os.cockpit.panel_types import CockpitPanel, PanelDefinition
from agent_core.creative_os.cockpit.panels import header_panel, issues_panel
from agent_core.creative_os.cockpit.state_adapter import CockpitState


def compose_layout(state: CockpitState) -> ComposeResult:
    with Container(id="app-root"):
        with Horizontal(id="cockpit-header"):
            yield Static(header_panel.render_brand(state), id="header-brand")
            yield Static(header_panel.render_details(state), id="header-details")
            yield Static(header_panel.render_meta(state), id="header-meta")
        with Horizontal(id="main-area"):
            with Vertical(id="sidebar"):
                yield _build_panel(PANEL_REGISTRY["system_status"], state)
                yield _build_panel(PANEL_REGISTRY["pipeline_map"], state)
            yield _build_panel(PANEL_REGISTRY["active_workspace"], state)
        with Horizontal(id="bottom-area"):
            yield _build_panel(PANEL_REGISTRY["skill_health"], state)
            yield _build_panel(PANEL_REGISTRY["artifacts"], state)
            with Vertical(id="right-bottom"):
                yield _build_panel(PANEL_REGISTRY["issues"], state)
                yield _build_panel(PANEL_REGISTRY["next"], state)
        yield Static(_keybar_text(state), id="help-panel")
        yield Static(_keybar_text(state), id="keybar")


def panel_update_targets() -> Iterator[tuple[str, PanelDefinition]]:
    yield "#header-brand", _HeaderPart("header_brand", header_panel.render_brand)
    yield "#header-details", _HeaderPart("header_details", header_panel.render_details)
    yield "#header-meta", _HeaderPart("header_meta", header_panel.render_meta)
    yield "#help-panel", _HeaderPart("help_panel", _keybar_text)
    yield "#keybar", _HeaderPart("keybar", _keybar_text)
    for panel in PANEL_REGISTRY.values():
        yield f"#{panel.widget_id}", panel


def _build_panel(panel: PanelDefinition, state: CockpitState) -> CockpitPanel:
    classes = panel.classes
    if panel.panel_id == "issues":
        classes = f"tile {issues_panel.severity_class(state)}"
    return CockpitPanel(panel.title, panel.render(state), panel_id=panel.widget_id, classes=classes)


class _HeaderPart:
    def __init__(self, panel_id: str, render: object) -> None:
        self.panel_id = panel_id
        self.render = render


def _keybar_text(state: CockpitState) -> str:
    watch = "on" if state.watch_enabled else "off"
    return f"q Quit · r Refresh · h Help · Watch {watch}"
