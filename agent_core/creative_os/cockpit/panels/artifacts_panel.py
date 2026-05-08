from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.state_adapter import CockpitState


def render(state: CockpitState) -> Text:
    artifacts = Text()
    for line, ok in state.artifacts.lines:
        artifacts.append(line + "\n", style="#22C55E" if ok else "#94A3B8")
    return artifacts
