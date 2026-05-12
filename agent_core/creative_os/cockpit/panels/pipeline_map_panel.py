from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import status_text_style
from agent_core.creative_os.cockpit.state_adapter import CockpitState
from agent_core.creative_os.cockpit.stage_registry import STAGE_DEFINITIONS, current_stage_id, selected_stage_id, status_for_stage
from agent_core.creative_os.cockpit.theme import TEXT_LABEL, style


def render(state: CockpitState) -> Text:
    timeline = Text()
    selected = selected_stage_id(state)
    current = current_stage_id(state)
    if selected == "08":
        current = selected
    for stage in STAGE_DEFINITIONS:
        status = status_for_stage(state, stage)
        if selected == "08" and stage.stage_id != selected and status == "current":
            status = "pending"
        marker = _marker(stage.stage_id, status, selected, current)
        line_style = _stage_style(stage.stage_id, status, selected, current)
        timeline.append(f"{marker} {stage.stage_id} {stage.title}\n", style=line_style)
    return timeline


def _marker(stage_id: str, status: str, selected: str, current: str) -> str:
    if stage_id == selected:
        return "▸"
    if stage_id == current:
        return "▶"
    if status == "passed":
        return "✓"
    if status in {"needs_review", "unknown", "current"}:
        return "!"
    if status == "rejected":
        return "✗"
    return "○"


def _stage_style(stage_id: str, status: str, selected: str, current: str) -> str:
    if stage_id == selected:
        return style(TEXT_LABEL, bold=True)
    if stage_id == current:
        return "bold #FBBF24"
    if status == "current":
        return "bold #FBBF24"
    return status_text_style(status if status != "current" else "unknown")
