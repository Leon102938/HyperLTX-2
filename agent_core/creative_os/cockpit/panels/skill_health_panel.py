from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.state_adapter import CockpitState


def render(state: CockpitState) -> Text:
    data = state.skill_health
    tile = Text()
    tile.append(f"{data.mark} {data.status}\n", style="bold #22C55E")
    tile.append(f"loaded {data.loaded_count} · fallbacks {data.fallback_count}\n", style="#E5E7EB")
    tile.append(f"missing optional {data.missing_optional_count} · blocking {data.blocking_missing_count}", style="#94A3B8")
    return tile
