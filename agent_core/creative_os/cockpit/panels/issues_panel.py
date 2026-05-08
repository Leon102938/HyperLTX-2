from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.state_adapter import CockpitState
from agent_core.creative_os.cockpit.theme import TEXT_ACTIVE, TEXT_ERROR, TEXT_SUCCESS


def render(state: CockpitState) -> Text:
    if not state.issues.blocking_issues:
        return Text("none blocking", style=f"bold {TEXT_SUCCESS}")
    style = f"bold {TEXT_ACTIVE}" if state.issues.severity == "warning" else f"bold {TEXT_ERROR}"
    return Text("\n".join(state.issues.blocking_issues), style=style)


def severity_class(state: CockpitState) -> str:
    if state.issues.severity == "warning":
        return "issues-warning"
    if state.issues.severity == "error":
        return "issues-error"
    return "issues-none"
