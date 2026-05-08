from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import status_text_style
from agent_core.creative_os.cockpit.state_adapter import CockpitState


def render(state: CockpitState) -> Text:
    timeline = Text()
    for stage in state.pipeline_map:
        mark = _mark(stage.status)
        prefix = "▶" if stage.index == "09" and stage.status == "pending" else mark
        style = "bold #FBBF24" if stage.index == "09" and stage.status == "pending" else status_text_style(stage.status)
        timeline.append(f"{prefix} {stage.index} {stage.name}\n", style=style)
    return timeline


def _mark(status: str) -> str:
    if status == "passed":
        return "✓"
    if status in {"needs_review", "unknown"}:
        return "!"
    if status == "rejected":
        return "✗"
    return "○"
