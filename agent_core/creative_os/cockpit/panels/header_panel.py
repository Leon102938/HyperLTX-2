from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import _shorten
from agent_core.creative_os.cockpit.state_adapter import CockpitState


def render_brand(_state: CockpitState) -> Text:
    title = Text()
    title.append("CONTENT MASCHINE ", style="bold #E5E7EB")
    title.append("LIVE\n", style="bold #FBBF24")
    title.append("CREATIVE OS COCKPIT", style="#67E8F9")
    return title


def render_details(state: CockpitState) -> Text:
    data = state.header
    mode = _shorten(data.mode + " · " + data.topic, 76)
    format_line = f"{data.orientation} · {data.resolution} · {data.duration}s · {data.scene_count} scenes"
    rows = [
        ("JOB", data.job_id),
        ("MODE", mode),
        ("FORMAT", format_line),
        ("PIPELINE", data.pipeline),
        ("STATUS", data.status),
        ("RENDER", data.render_state),
        ("FOCUS", "CLI cockpit refinement"),
    ]
    details = Text()
    for label, value in rows:
        details.append(f"{label:<9}  ", style="bold #67E8F9")
        details.append(_shorten(str(value), 76), style="#E5E7EB")
        details.append("\n")
    return details


def render_meta(state: CockpitState) -> Text:
    data = state.header
    meta = Text()
    rows = (("SESSION", data.session), ("RUN TYPE", data.run_type), ("WATCH", data.watch))
    for index, (label, value) in enumerate(rows):
        if index:
            meta.append("\n")
        meta.append(f"{label}\n", style="bold #67E8F9")
        meta.append(value, style="#94A3B8")
        if index < len(rows) - 1:
            meta.append("\n")
    return meta
