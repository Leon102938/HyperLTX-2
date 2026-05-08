from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import rows_text
from agent_core.creative_os.cockpit.state_adapter import CockpitState


def render(state: CockpitState) -> Text:
    return rows_text(state.next_panel.rows, label_width=10)
