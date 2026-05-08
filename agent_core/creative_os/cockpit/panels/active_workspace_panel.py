from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import rows_text, scene_card_text
from agent_core.creative_os.cockpit.state_adapter import CockpitState
from agent_core.creative_os.cockpit.theme import BG_WORKSPACE, TEXT_LABEL, TEXT_MUTED, TEXT_SUCCESS, style


def render(state: CockpitState) -> Text:
    if not state.run_found:
        return _missing_run_text(state)

    data = state.workspace
    workspace = rows_text(
        [
            ("Current Step", data.current_step),
            ("Last passed", data.last_passed),
            ("Next technical", data.next_technical),
            ("Operator focus", data.operator_focus),
            ("Render paused", data.render_paused),
        ],
        bg=BG_WORKSPACE,
    )
    workspace.append("\nSTAGE OUTPUT  ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    if state.run_type == "agent_core":
        workspace.append(f"{state.header.run_type}  ", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append(f"{state.header.scene_count} scenes  ", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append(f"final.mp4 {state.header.render_state}\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    else:
        workspace.append("✓ 3 motion prompts  ", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append("✓ 3 keyframes  ", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append("○ video takes not built\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    workspace.append("SCENE JOBS\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    if not data.scenes:
        workspace.append("No scene jobs available\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    else:
        for scene in data.scenes:
            workspace.append(scene_card_text(scene))
    return workspace


def _missing_run_text(state: CockpitState) -> Text:
    workspace = rows_text(
        [
            ("Current Step", "Run not found"),
            ("Searched", str(state.data_source_path)),
            ("Hint", "use --runs-root for fixture/demo data or create a real run first"),
            ("Watch", "on" if state.watch_enabled else "off"),
            ("Render paused", "unknown"),
        ],
        bg=BG_WORKSPACE,
    )
    workspace.append("\nRun not found\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(f"searched: {state.data_source_path}\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    workspace.append(
        "hint: use --runs-root for fixture/demo data or create a real run first\n",
        style=style(TEXT_MUTED, bg=BG_WORKSPACE),
    )
    return workspace
