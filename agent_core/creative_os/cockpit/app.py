from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Static

from agent_core.creative_os.cockpit.layout import compose_layout, panel_update_targets
from agent_core.creative_os.cockpit.panels import issues_panel
from agent_core.creative_os.cockpit.stage_registry import DEFAULT_STAGE_ID, normalize_stage_id, stage_ids
from agent_core.creative_os.cockpit.state_adapter import CockpitStateAdapter
from agent_core.creative_os.cockpit.theme import (
    BG_PANEL,
    BG_SCENE_CARD,
    BG_WORKSPACE,
    BORDER_PRIMARY,
    BORDER_SECONDARY,
    COCKPIT_CSS,
    TEXT_ACTIVE,
    TEXT_ERROR,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
    TEXT_SUCCESS,
    style,
)


@dataclass(frozen=True)
class CockpitArgs:
    job_id: str
    runs_root: Path
    watch: bool = False
    refresh_sec: float = 2.0


class CreativeOSCockpitApp(App[None]):
    CSS = COCKPIT_CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("h", "toggle_help", "Help", show=True),
        Binding("down", "select_next_stage", "Next Stage", show=False),
        Binding("up", "select_previous_stage", "Prev Stage", show=False),
        Binding("j", "select_next_image_job", "Next Image Job", show=False),
        Binding("k", "select_previous_image_job", "Prev Image Job", show=False),
        Binding("enter,space", "open_selected_stage", "Open Stage", show=False),
    ]

    def __init__(self, args: CockpitArgs) -> None:
        super().__init__()
        self.args = args
        self.state_adapter = CockpitStateAdapter(
            job_id=args.job_id,
            runs_root=args.runs_root,
            watch_enabled=args.watch,
            refresh_sec=args.refresh_sec,
        )
        self.selected_stage = DEFAULT_STAGE_ID
        self.selected_image_job = 2
        self.expanded_image_jobs = (2,)
        self.state = self._state_with_selected_stage(self.state_adapter.load())
        self.inspector = self.state_adapter.inspector
        self.inspection = self.state.inspection

    def compose(self) -> ComposeResult:
        yield from compose_layout(self.state)

    def on_mount(self) -> None:
        if self.args.watch:
            self.set_interval(self.args.refresh_sec, self._watch_refresh)

    def action_quit(self) -> None:
        self.exit()

    def action_refresh(self) -> None:
        self._reload_state()
        self.notify("Cockpit data reloaded", timeout=1.5)

    def action_toggle_help(self) -> None:
        help_panel = self.query_one("#help-panel", Static)
        help_panel.toggle_class("visible")

    def action_select_next_stage(self) -> None:
        self._move_selected_stage(1)

    def action_select_previous_stage(self) -> None:
        self._move_selected_stage(-1)

    def action_open_selected_stage(self) -> None:
        if self.selected_stage == "09":
            self._toggle_selected_image_job()
            return
        self._update_panels()

    def action_select_next_image_job(self) -> None:
        if self.selected_stage == "09":
            self._move_selected_image_job(1)
            return
        self._move_selected_stage(1)

    def action_select_previous_image_job(self) -> None:
        if self.selected_stage == "09":
            self._move_selected_image_job(-1)
            return
        self._move_selected_stage(-1)

    def on_click(self, event: object) -> None:
        widget = getattr(event, "widget", None)
        if getattr(widget, "id", None) != "pipeline-map":
            return
        y = int(getattr(event, "y", -1))
        stage_index = y - 1
        ids = stage_ids()
        if 0 <= stage_index < len(ids):
            self._select_stage(ids[stage_index])

    def _update_panels(self) -> None:
        for selector, panel in panel_update_targets():
            self.query_one(selector, Static).update(panel.render(self.state))
        issues_tile = self.query_one("#issues-tile", Static)
        issues_tile.remove_class("issues-none", "issues-warning", "issues-error")
        issues_tile.add_class(issues_panel.severity_class(self.state))

    def _update_selection_panels(self) -> None:
        selection_selectors = {"#pipeline-map", "#workspace-content"}
        for selector, panel in panel_update_targets():
            if selector in selection_selectors:
                self.query_one(selector, Static).update(panel.render(self.state))

    def _watch_refresh(self) -> None:
        loaded_state = self._state_with_selected_stage(self.state_adapter.load())
        if loaded_state == self.state:
            return
        self.state = loaded_state
        self.inspection = self.state.inspection
        self._update_panels()

    def _reload_state(self) -> None:
        self.state = self._state_with_selected_stage(self.state_adapter.load())
        self.inspection = self.state.inspection
        self._update_panels()

    def _move_selected_stage(self, offset: int) -> None:
        ids = stage_ids()
        current = ids.index(normalize_stage_id(self.selected_stage))
        self._select_stage(ids[(current + offset) % len(ids)])

    def _select_stage(self, stage_id: str) -> None:
        self.selected_stage = normalize_stage_id(stage_id)
        self.state = self._state_with_selected_stage(self.state)
        self._update_selection_panels()

    def _state_with_selected_stage(self, state: object) -> object:
        return replace(
            state,
            selected_stage=normalize_stage_id(self.selected_stage),
            selected_image_job=self.selected_image_job,
            expanded_image_jobs=self.expanded_image_jobs,
        )

    def _move_selected_image_job(self, offset: int) -> None:
        self.selected_image_job = ((self.selected_image_job - 1 + offset) % 3) + 1
        self.state = self._state_with_selected_stage(self.state)
        self._update_selection_panels()

    def _toggle_selected_image_job(self) -> None:
        expanded = set(self.expanded_image_jobs)
        if self.selected_image_job in expanded:
            expanded.remove(self.selected_image_job)
        else:
            expanded.add(self.selected_image_job)
        self.expanded_image_jobs = tuple(sorted(expanded))
        self.state = self._state_with_selected_stage(self.state)
        self._update_selection_panels()


class ThemePreviewApp(App[None]):
    CSS = COCKPIT_CSS

    def compose(self) -> ComposeResult:
        with Container(id="theme-preview-root"):
            yield Static(_theme_preview_header(), id="theme-preview-header")
            with Horizontal(id="theme-preview-grid"):
                yield Static(_theme_preview_panel(), id="theme-preview-primary", classes="theme-preview-panel")
                yield Static(_theme_preview_workspace(), id="theme-preview-workspace", classes="theme-preview-panel")
                yield Static(_theme_preview_secondary(), id="theme-preview-secondary", classes="theme-preview-panel")
            yield Static(_theme_preview_footer(), id="theme-preview-footer")


def _theme_preview_header() -> Text:
    text = Text()
    text.append("THEME PREVIEW\n", style=style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("App Background #050B12 · Panel Background #07111F · Workspace #0B1628", style=style(TEXT_MAIN, bg=BG_PANEL))
    return text


def _theme_preview_panel() -> Text:
    text = Text()
    text.append("PANEL BACKGROUND\n", style=style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("primary border #38BDF8\n", style=style(BORDER_PRIMARY, bg=BG_PANEL))
    text.append("main text #E5E7EB\n", style=style(TEXT_MAIN, bg=BG_PANEL))
    text.append("label #67E8F9\n", style=style(TEXT_LABEL, bg=BG_PANEL))
    text.append("muted #64748B", style=style(TEXT_MUTED, bg=BG_PANEL))
    return text


def _theme_preview_workspace() -> Text:
    text = Text()
    text.append("WORKSPACE BACKGROUND\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    text.append("Scene Card Sample\n", style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    text.append("╭────────────────────────────╮\n", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("│ ", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("scene-card #071827".ljust(27), style=style(TEXT_MAIN, bg=BG_SCENE_CARD))
    text.append("│\n", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("│ ", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("active #FBBF24".ljust(27), style=style(TEXT_ACTIVE, bg=BG_SCENE_CARD, bold=True))
    text.append("│\n", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("╰────────────────────────────╯", style=style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    return text


def _theme_preview_secondary() -> Text:
    text = Text()
    text.append("STATUS COLORS\n", style=style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("✓ success #22C55E\n", style=style(TEXT_SUCCESS, bg=BG_PANEL, bold=True))
    text.append("▶ active #FBBF24\n", style=style(TEXT_ACTIVE, bg=BG_PANEL, bold=True))
    text.append("○ pending #64748B\n", style=style(TEXT_MUTED, bg=BG_PANEL))
    text.append("✗ error #EF4444\n", style=style(TEXT_ERROR, bg=BG_PANEL, bold=True))
    text.append("secondary border #1E3A5F", style=style(BORDER_SECONDARY, bg=BG_PANEL))
    return text


def _theme_preview_footer() -> Text:
    text = Text()
    text.append("Visible background check: ", style=style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("header, panels, workspace, scene-card and bottom band all carry explicit dark-blue backgrounds.", style=style(TEXT_MAIN, bg=BG_PANEL))
    return text


def run_cockpit(job_id: str, runs_root: str | Path, *, watch: bool = False, refresh_sec: float = 2.0) -> None:
    CreativeOSCockpitApp(CockpitArgs(job_id=job_id, runs_root=Path(runs_root), watch=watch, refresh_sec=refresh_sec)).run()


def run_theme_preview() -> None:
    ThemePreviewApp().run()
