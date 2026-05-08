from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .run_inspector import CreativeOSRunInspector, RunInspection


try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import Static
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - exercised by script import guard.
    raise RuntimeError(
        "Textual is not installed. Install dependencies with: python3 -m pip install 'textual>=0.89,<1.0'"
    ) from exc


FIXTURE_MARKER = "tests/fixtures/creative_os_runs"
BG_SCREEN = "#050B12"
BG_PANEL = "#07111F"
BG_WORKSPACE = "#0B1628"
BG_SCENE_CARD = "#071827"
BORDER_PRIMARY = "#38BDF8"
BORDER_SECONDARY = "#1E3A5F"
TEXT_MAIN = "#E5E7EB"
TEXT_LABEL = "#67E8F9"
TEXT_MUTED = "#64748B"
TEXT_SUCCESS = "#22C55E"
TEXT_ACTIVE = "#FBBF24"
TEXT_ERROR = "#EF4444"


@dataclass(frozen=True)
class CockpitArgs:
    job_id: str
    runs_root: Path


class CockpitPanel(Static):
    def __init__(self, title: str, body: str, *, panel_id: str | None = None, classes: str | None = None) -> None:
        super().__init__(body, id=panel_id, classes=classes)
        self.border_title = title


class CreativeOSCockpitApp(App[None]):
    CSS = """
    Screen {
        background: #050B12;
        color: #E5E7EB;
        width: 100%;
        height: 100%;
    }

    Container,
    Horizontal,
    Vertical {
        background: #050B12;
        color: #E5E7EB;
        width: 100%;
    }

    #app-root {
        height: 100%;
        width: 100%;
        padding: 1 1 0 1;
        background: #050B12;
        color: #E5E7EB;
    }

    .panel {
        background: #07111F;
        border: round #38BDF8;
        padding: 1 2;
        margin: 0 1 1 0;
        color: #E5E7EB;
    }

    #cockpit-header {
        height: 12;
        width: 100%;
        border: round #38BDF8;
        background: #07111F;
        color: #E5E7EB;
        padding: 1 2;
        margin-bottom: 1;
    }

    #header-brand {
        width: 42%;
        height: 100%;
        content-align: center middle;
        color: #38BDF8;
        text-style: bold;
        background: #0B1628;
        border-right: solid #38BDF8;
    }

    #header-details {
        width: 1fr;
        padding-left: 2;
        color: #E5E7EB;
        text-style: bold;
        background: #07111F;
    }

    #header-meta {
        width: 27;
        padding-left: 2;
        border-left: solid #38BDF8;
        color: #94A3B8;
        background: #07111F;
    }

    #main-area {
        width: 100%;
        height: 1fr;
        min-height: 13;
        margin-bottom: 1;
        background: #050B12;
    }

    #sidebar {
        width: 36%;
        min-width: 36;
        margin-right: 1;
        background: #07111F;
    }

    #workspace {
        width: 1fr;
        border: round #38BDF8;
        background: #0B1628;
        color: #E5E7EB;
        padding: 1 2;
        height: 100%;
        overflow: hidden hidden;
    }

    #system-status {
        height: 11;
        border: round #38BDF8;
    }

    #pipeline-map {
        border: round #1E3A5F;
        height: 1fr;
        overflow: hidden hidden;
    }

    #bottom-area {
        width: 100%;
        height: 9;
        margin-bottom: 0;
        background: #050B12;
    }

    .tile {
        width: 1fr;
        min-width: 24;
        background: #07111F;
        border: round #38BDF8;
        color: #E5E7EB;
        padding: 1 2;
        margin-right: 1;
        height: 100%;
    }

    #skill-tile {
        width: 33%;
        border: round #38BDF8;
    }

    #artifacts-tile {
        width: 33%;
        border: round #1E3A5F;
    }

    #right-bottom {
        width: 30%;
        background: #07111F;
    }

    #issues-tile,
    #next-tile {
        width: 100%;
        height: 1fr;
        margin-right: 0;
    }

    #next-tile {
        border: round #38BDF8;
    }

    #issues-tile.ok {
        border: round #22C55E;
    }

    #help-panel {
        dock: bottom;
        height: 5;
        display: none;
        border: round #38BDF8;
        background: #07111F;
        padding: 1 2;
        margin: 1;
    }

    #help-panel.visible {
        display: block;
    }

    .title {
        color: #67E8F9;
        text-style: bold;
    }

    .ok {
        color: #22C55E;
    }

    .active {
        color: #FBBF24;
        text-style: bold;
    }

    .muted {
        color: #64748B;
    }

    .label {
        color: #67E8F9;
    }

    #keybar {
        height: 1;
        width: 100%;
        background: #050B12;
        color: #38BDF8;
        content-align: center middle;
        margin-top: 0;
    }

    #theme-preview-root {
        width: 100%;
        height: 100%;
        padding: 1 2;
        background: #050B12;
        color: #E5E7EB;
    }

    #theme-preview-header {
        height: 5;
        width: 100%;
        background: #07111F;
        border: round #38BDF8;
        padding: 1 2;
        margin-bottom: 1;
        color: #E5E7EB;
    }

    #theme-preview-grid {
        height: 1fr;
        width: 100%;
        background: #050B12;
    }

    .theme-preview-panel {
        width: 1fr;
        height: 100%;
        margin-right: 1;
        padding: 1 2;
        background: #07111F;
        border: round #38BDF8;
        color: #E5E7EB;
    }

    #theme-preview-workspace {
        background: #0B1628;
    }

    #theme-preview-secondary {
        border: round #1E3A5F;
    }

    #theme-preview-footer {
        height: 5;
        width: 100%;
        margin-top: 1;
        padding: 1 2;
        background: #07111F;
        border: round #1E3A5F;
        color: #E5E7EB;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("h", "toggle_help", "Help", show=True),
    ]

    def __init__(self, args: CockpitArgs) -> None:
        super().__init__()
        self.args = args
        self.inspector = CreativeOSRunInspector(runs_root=args.runs_root)
        self.inspection = self.inspector.inspect(args.job_id)

    def compose(self) -> ComposeResult:
        with Container(id="app-root"):
            with Horizontal(id="cockpit-header"):
                yield Static(self._brand_title(), id="header-brand")
                yield Static(self._header_details(), id="header-details")
                yield Static(self._header_meta(), id="header-meta")
            with Horizontal(id="main-area"):
                with Vertical(id="sidebar"):
                    yield CockpitPanel("SYSTEM STATUS", self._system_status(), panel_id="system-status", classes="panel")
                    yield CockpitPanel("PIPELINE MAP", self._pipeline_map(), panel_id="pipeline-map", classes="panel")
                yield CockpitPanel("ACTIVE WORKSPACE", self._workspace(), panel_id="workspace")
            with Horizontal(id="bottom-area"):
                yield CockpitPanel("SKILL HEALTH", self._skill_health(), panel_id="skill-tile", classes="tile")
                yield CockpitPanel("ARTIFACTS", self._artifacts(), panel_id="artifacts-tile", classes="tile")
                with Vertical(id="right-bottom"):
                    yield CockpitPanel("ISSUES", self._issues(), panel_id="issues-tile", classes="tile ok")
                    yield CockpitPanel("NEXT", self._next(), panel_id="next-tile", classes="tile")
            yield Static("q Quit · r Refresh · h Help", id="help-panel")
            yield Static("q Quit · r Refresh · h Help", id="keybar")

    def action_refresh(self) -> None:
        self.inspection = self.inspector.inspect(self.args.job_id)
        self._update_panels()
        self.notify("Fixture data reloaded", timeout=1.5)

    def action_toggle_help(self) -> None:
        help_panel = self.query_one("#help-panel", Static)
        help_panel.toggle_class("visible")

    def _update_panels(self) -> None:
        updates = {
            "#header-details": self._header_details(),
            "#header-meta": self._header_meta(),
            "#system-status": self._system_status(),
            "#pipeline-map": self._pipeline_map(),
            "#workspace": self._workspace(),
            "#skill-tile": self._skill_health(),
            "#artifacts-tile": self._artifacts(),
            "#issues-tile": self._issues(),
            "#next-tile": self._next(),
        }
        for selector, content in updates.items():
            self.query_one(selector, Static).update(content)

    def _header_details(self) -> Text:
        meta = _job_meta(self.inspection)
        rows = [
            (("JOB", str(self.inspection.job_id)), ("PIPELINE", "shortform_storyboard_v1")),
            None,
            (("MODE", _shorten(meta["mode"] + " · " + meta["topic"], 78)),),
            (("FORMAT", f"{meta['orientation']} · {meta['resolution']} · {meta['duration']}s · {meta['scene_count']} scenes"),),
            None,
            (("STATUS", self.inspection.status), ("FOCUS", "CLI cockpit refinement")),
            (("RENDER", "paused"),),
        ]
        details = Text()
        for row in rows:
            if row is None:
                details.append("\n")
                continue
            for index, (label, value) in enumerate(row):
                if index:
                    details.append("     ")
                details.append(f"{label:<9}", style="bold #67E8F9")
                details.append(f"{str(value):<36}", style="#E5E7EB")
            details.append("\n")
        return details

    def _brand_title(self) -> Text:
        title = Text()
        title.append("▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜\n", style="#38BDF8")
        title.append("▌                               ▐\n", style="#38BDF8")
        title.append("  CONTENT  MASCHINE  ", style="bold #E5E7EB")
        title.append("LIVE\n", style="bold #FBBF24")
        title.append("▌                               ▐\n", style="#38BDF8")
        title.append("▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟", style="#38BDF8")
        return title

    def _header_meta(self) -> Text:
        meta = Text()
        rows = [
            ("SESSION", _session_label(self.inspection)),
            ("CHECKS", "no live checks"),
            ("MODE", "fixture" if FIXTURE_MARKER in str(self.inspection.run_dir) else "artifact read"),
        ]
        for index, (label, value) in enumerate(rows):
            if index:
                meta.append("\n")
            meta.append(f"{label}\n", style="bold #67E8F9")
            meta.append(value, style="#94A3B8")
            if index < len(rows) - 1:
                meta.append("\n")
        return meta

    def _system_status(self) -> Text:
        vision = "- manual_structured" if "stage6_review_decision" in self.inspection.data else "- heuristic"
        image = "✓ zimage_http" if self.inspection.artifacts.get("keyframe_manifest.json") else "? not_checked"
        return _rows_text(
            [
                ("API", "? not_checked"),
                ("Director", "? not_checked"),
                ("Image Backend", image),
                ("Video Backend", "? planned ltx2"),
                ("Vision Review", vision),
                ("Voice", "- disabled"),
                ("Music", "- disabled"),
                ("Subtitles", "- off"),
            ]
        )

    def _pipeline_map(self) -> Text:
        timeline = Text()
        for stage in self.inspection.stages:
            mark = _mark(stage.status)
            prefix = "▶" if stage.index == "09" and stage.status == "pending" else mark
            style = "bold #FBBF24" if stage.index == "09" and stage.status == "pending" else _status_text_style(stage.status)
            timeline.append(f"{prefix} {stage.index} {stage.name}\n", style=style)
        return timeline

    def _workspace(self) -> Text:
        workspace = _rows_text(
            [
                ("Current Step", "LTX Motion Ready"),
                ("Last passed", f"✓ {self.inspector.last_passed_stage(self.inspection)}"),
                ("Next technical", "○ 09 LTX I2V takes"),
                ("Operator focus", "CLI cockpit refinement"),
                ("Render paused", "yes"),
            ],
            bg=BG_WORKSPACE,
        )
        workspace.append("\nSTAGE OUTPUT\n", style=_style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        workspace.append("  ✓ 3 motion prompts passed\n", style=_style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append("  ✓ 3 keyframes passed\n", style=_style(TEXT_SUCCESS, bg=BG_WORKSPACE))
        workspace.append("  ○ video takes not built\n", style=_style(TEXT_MUTED, bg=BG_WORKSPACE))
        workspace.append("\nSCENE JOBS\n", style=_style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        for item in _motion_items(self.inspection):
            workspace.append("\n", style=_style(TEXT_MAIN, bg=BG_WORKSPACE))
            workspace.append("╭────────────────────────────────────────────────────────────╮\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append("│", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append(f" [✓] {item['scene_id']}  READY FOR LTX".ljust(60), style=_style(TEXT_SUCCESS, bg=BG_SCENE_CARD, bold=True))
            workspace.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append("│", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append(f" keyframe: {_basename(item['keyframe'])}".ljust(60), style=_style(TEXT_MAIN, bg=BG_SCENE_CARD))
            workspace.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append("│", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append(f" motion:   {item['summary']}".ljust(60), style=_style(TEXT_MUTED, bg=BG_SCENE_CARD))
            workspace.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append("│", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append(" status:   ", style=_style(TEXT_MAIN, bg=BG_SCENE_CARD))
            workspace.append("waiting for Stage 09 render gate".ljust(49), style=_style(TEXT_ACTIVE, bg=BG_SCENE_CARD))
            workspace.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
            workspace.append("╰────────────────────────────────────────────────────────────╯\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
        return workspace

    def _skill_health(self) -> Text:
        health = self.inspector.skill_health(self.inspection)
        tile = Text()
        tile.append(f"{health['mark']} {health['status']}\n", style="bold #22C55E")
        tile.append(f"loaded {len(health['loaded'])} · fallbacks {len(health['fallbacks'])}\n", style="#E5E7EB")
        tile.append(f"missing optional {len(health['missing_optional'])} · blocking {len(health['blocking_missing'])}", style="#94A3B8")
        return tile

    def _artifacts(self) -> Text:
        keyframes = sum(
            1 for scene_id in ("scene_01", "scene_02", "scene_03") if self.inspection.artifacts.get(f"keyframes/{scene_id}.png")
        )
        artifacts = Text()
        for line in [
            f"{_artifact_mark(self.inspection, 'zimage_prompts.json')} zimage_prompts.json",
            f"{'✓' if keyframes == 3 else '○'} {keyframes} keyframes",
            f"{_artifact_mark(self.inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
            f"{_artifact_mark(self.inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
            f"{_artifact_mark(self.inspection, 'ltx_video_takes_manifest.json')} video_takes_manifest",
        ]:
            artifacts.append(line + "\n", style="#22C55E" if line.startswith("✓") else "#94A3B8")
        return artifacts

    def _issues(self) -> Text:
        return Text("none blocking", style="bold #22C55E") if not self.inspection.blocking_issues else Text("\n".join(self.inspection.blocking_issues), style="bold #EF4444")

    def _next(self) -> Text:
        return _rows_text([("Technical", "Stage 09: LTX I2V takes"), ("Operator", "Improve CLI cockpit")], label_width=10)


def _row(label: str, value: str) -> str:
    return f"{label.upper():<16} {value}"


class ThemePreviewApp(App[None]):
    CSS = CreativeOSCockpitApp.CSS

    def compose(self) -> ComposeResult:
        with Container(id="theme-preview-root"):
            yield Static(_theme_preview_header(), id="theme-preview-header")
            with Horizontal(id="theme-preview-grid"):
                yield Static(_theme_preview_panel(), id="theme-preview-primary", classes="theme-preview-panel")
                yield Static(_theme_preview_workspace(), id="theme-preview-workspace", classes="theme-preview-panel")
                yield Static(_theme_preview_secondary(), id="theme-preview-secondary", classes="theme-preview-panel")
            yield Static(_theme_preview_footer(), id="theme-preview-footer")


def _rows_text(rows: list[tuple[str, str]], *, label_width: int = 16, bg: str | None = None) -> Text:
    text = Text()
    for label, value in rows:
        text.append(f"{label.upper():<{label_width}} ", style=_style(TEXT_LABEL, bg=bg, bold=True))
        text.append(f"{value}\n", style=_value_style(value, bg=bg))
    return text


def _style(color: str, *, bg: str | None = None, bold: bool = False) -> str:
    parts = []
    if bold:
        parts.append("bold")
    parts.append(color)
    if bg:
        parts.append(f"on {bg}")
    return " ".join(parts)


def _value_style(value: str, *, bg: str | None = None) -> str:
    if value.startswith("✓"):
        return _style(TEXT_SUCCESS, bg=bg, bold=True)
    if value.startswith("?") or value.startswith("-"):
        return _style(TEXT_MUTED, bg=bg)
    if "Stage 09" in value or value == "paused":
        return _style(TEXT_ACTIVE, bg=bg, bold=True)
    return _style(TEXT_MAIN, bg=bg)


def _status_text_style(status: str) -> str:
    if status == "passed":
        return "#22C55E"
    if status == "pending":
        return "#94A3B8"
    if status in {"needs_review", "unknown"}:
        return "#FBBF24"
    if status == "rejected":
        return "#EF4444"
    return "#E5E7EB"


def _mark(status: str) -> str:
    if status == "passed":
        return "✓"
    if status in {"needs_review", "unknown"}:
        return "!"
    if status == "rejected":
        return "✗"
    return "○"


def _artifact_mark(inspection: RunInspection, name: str) -> str:
    return "✓" if inspection.artifacts.get(name) else "○"


def _session_label(inspection: RunInspection) -> str:
    return "fixture/demo" if FIXTURE_MARKER in str(inspection.run_dir) else "artifact read"


def _job_meta(inspection: RunInspection) -> dict[str, object]:
    job = inspection.data.get("normalized_job") or {}
    route = inspection.data.get("intent_route") or {}
    scenes = inspection.data.get("scene_contracts") or []
    return {
        "resolution": job.get("resolution") or "512x768",
        "duration": job.get("duration_sec") or 9,
        "orientation": job.get("orientation") or "portrait",
        "mode": route.get("genre_intent") or route.get("mode_intent") or "unknown",
        "topic": route.get("topic_intent") or "unknown",
        "scene_count": len(scenes) or 3,
    }


def _motion_items(inspection: RunInspection) -> list[dict[str, str]]:
    prompts = inspection.data.get("ltx_motion_prompts") or []
    if not isinstance(prompts, list):
        return []
    items: list[dict[str, str]] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        keyframe = str(prompt.get("source_keyframe_path") or "")
        items.append(
            {
                "scene_id": str(prompt.get("scene_id") or "unknown"),
                "keyframe": keyframe,
                "summary": _shorten(str(prompt.get("camera_motion") or prompt.get("motion_prompt") or "motion prompt present"), 65),
            }
        )
    return items


def _shorten(text: object, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else "missing"


def _theme_preview_header() -> Text:
    text = Text()
    text.append("THEME PREVIEW\n", style=_style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("App Background #050B12 · Panel Background #07111F · Workspace #0B1628", style=_style(TEXT_MAIN, bg=BG_PANEL))
    return text


def _theme_preview_panel() -> Text:
    text = Text()
    text.append("PANEL BACKGROUND\n", style=_style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("primary border #38BDF8\n", style=_style(BORDER_PRIMARY, bg=BG_PANEL))
    text.append("main text #E5E7EB\n", style=_style(TEXT_MAIN, bg=BG_PANEL))
    text.append("label #67E8F9\n", style=_style(TEXT_LABEL, bg=BG_PANEL))
    text.append("muted #64748B", style=_style(TEXT_MUTED, bg=BG_PANEL))
    return text


def _theme_preview_workspace() -> Text:
    text = Text()
    text.append("WORKSPACE BACKGROUND\n", style=_style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    text.append("Scene Card Sample\n", style=_style(TEXT_MAIN, bg=BG_WORKSPACE))
    text.append("╭────────────────────────────╮\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("│ ", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("scene-card #071827".ljust(27), style=_style(TEXT_MAIN, bg=BG_SCENE_CARD))
    text.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("│ ", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("active #FBBF24".ljust(27), style=_style(TEXT_ACTIVE, bg=BG_SCENE_CARD, bold=True))
    text.append("│\n", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    text.append("╰────────────────────────────╯", style=_style(BORDER_PRIMARY, bg=BG_SCENE_CARD))
    return text


def _theme_preview_secondary() -> Text:
    text = Text()
    text.append("STATUS COLORS\n", style=_style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("✓ success #22C55E\n", style=_style(TEXT_SUCCESS, bg=BG_PANEL, bold=True))
    text.append("▶ active #FBBF24\n", style=_style(TEXT_ACTIVE, bg=BG_PANEL, bold=True))
    text.append("○ pending #64748B\n", style=_style(TEXT_MUTED, bg=BG_PANEL))
    text.append("✗ error #EF4444\n", style=_style(TEXT_ERROR, bg=BG_PANEL, bold=True))
    text.append("secondary border #1E3A5F", style=_style(BORDER_SECONDARY, bg=BG_PANEL))
    return text


def _theme_preview_footer() -> Text:
    text = Text()
    text.append("Visible background check: ", style=_style(TEXT_LABEL, bg=BG_PANEL, bold=True))
    text.append("header, panels, workspace, scene-card and bottom band all carry explicit dark-blue backgrounds.", style=_style(TEXT_MAIN, bg=BG_PANEL))
    return text


def run_cockpit(job_id: str, runs_root: str | Path) -> None:
    CreativeOSCockpitApp(CockpitArgs(job_id=job_id, runs_root=Path(runs_root))).run()


def run_theme_preview() -> None:
    ThemePreviewApp().run()
