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
        background: #06111A;
        color: #E8F4FF;
    }

    #app-root {
        height: 100%;
        padding: 1 1 0 1;
        background: #06111A;
    }

    .panel {
        background: #081722;
        border: round #16C6FF;
        padding: 1 2;
        margin: 0 1 1 0;
        color: #E8F4FF;
    }

    #cockpit-header {
        height: 12;
        border: round #16C6FF;
        background: #081722;
        padding: 1 2;
        margin-bottom: 1;
    }

    #header-brand {
        width: 42%;
        height: 100%;
        content-align: center middle;
        color: #16C6FF;
        text-style: bold;
        background: #06111A;
        border-right: solid #16C6FF;
    }

    #header-details {
        width: 1fr;
        padding-left: 2;
        color: #DFF0FF;
        text-style: bold;
    }

    #header-meta {
        width: 27;
        padding-left: 2;
        border-left: solid #16C6FF;
        color: #A8C7DB;
    }

    #main-area {
        height: 1fr;
        min-height: 13;
        margin-bottom: 1;
    }

    #sidebar {
        width: 36%;
        min-width: 36;
        margin-right: 1;
    }

    #workspace {
        width: 1fr;
        border: round #16C6FF;
        background: #0B1E2D;
        padding: 1 2;
        height: 100%;
        overflow: hidden hidden;
    }

    #system-status {
        height: 11;
        border: round #16C6FF;
    }

    #pipeline-map {
        height: 1fr;
        overflow: hidden hidden;
    }

    #bottom-area {
        height: 9;
        margin-bottom: 0;
    }

    .tile {
        width: 1fr;
        min-width: 24;
        background: #081722;
        border: round #16C6FF;
        padding: 1 2;
        margin-right: 1;
        height: 100%;
    }

    #skill-tile {
        width: 33%;
        border: round #16C6FF;
    }

    #artifacts-tile {
        width: 33%;
    }

    #right-bottom {
        width: 30%;
    }

    #issues-tile,
    #next-tile {
        width: 100%;
        height: 1fr;
        margin-right: 0;
    }

    #next-tile {
        border: round #16C6FF;
    }

    #issues-tile.ok {
        border: round #22C55E;
    }

    #help-panel {
        dock: bottom;
        height: 5;
        display: none;
        border: round #16C6FF;
        background: #091923;
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
        background: #06111A;
        color: #16C6FF;
        content-align: center middle;
        margin-top: 0;
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
                details.append(f"{label:<9}", style="bold #16C6FF")
                details.append(f"{str(value):<36}", style="#E8F4FF")
            details.append("\n")
        return details

    def _brand_title(self) -> Text:
        title = Text()
        title.append("▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▜\n", style="#16C6FF")
        title.append("▌                               ▐\n", style="#16C6FF")
        title.append("  CONTENT  MASCHINE  ", style="bold #E8F4FF")
        title.append("LIVE\n", style="bold #FFB347")
        title.append("▌                               ▐\n", style="#16C6FF")
        title.append("▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟", style="#16C6FF")
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
            meta.append(f"{label}\n", style="bold #16C6FF")
            meta.append(value, style="#B7D2E6")
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
            style = "bold #FFB347" if stage.index == "09" and stage.status == "pending" else _status_text_style(stage.status)
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
            ]
        )
        workspace.append("\nSTAGE OUTPUT\n", style="bold #16C6FF")
        workspace.append("  ✓ 3 motion prompts passed\n", style="#7CD66B")
        workspace.append("  ✓ 3 keyframes passed\n", style="#7CD66B")
        workspace.append("  ○ video takes not built\n", style="#A8C7DB")
        workspace.append("\nSCENE JOBS\n", style="bold #16C6FF")
        for item in _motion_items(self.inspection):
            workspace.append("\n╭────────────────────────────────────────────────────────────╮\n", style="#16C6FF")
            workspace.append("│", style="#16C6FF")
            workspace.append(f" [✓] {item['scene_id']}  READY FOR LTX".ljust(60), style="bold #7CD66B")
            workspace.append("│\n", style="#16C6FF")
            workspace.append("│", style="#16C6FF")
            workspace.append(f" keyframe: {_basename(item['keyframe'])}".ljust(60), style="#E8F4FF")
            workspace.append("│\n", style="#16C6FF")
            workspace.append("│", style="#16C6FF")
            workspace.append(f" motion:   {item['summary']}".ljust(60), style="#B7D2E6")
            workspace.append("│\n", style="#16C6FF")
            workspace.append("│", style="#16C6FF")
            workspace.append(" status:   ", style="#E8F4FF")
            workspace.append("waiting for Stage 09 render gate".ljust(49), style="#FFB347")
            workspace.append("│\n", style="#16C6FF")
            workspace.append("╰────────────────────────────────────────────────────────────╯\n", style="#16C6FF")
        return workspace

    def _skill_health(self) -> Text:
        health = self.inspector.skill_health(self.inspection)
        tile = Text()
        tile.append(f"{health['mark']} {health['status']}\n", style="bold #7CD66B")
        tile.append(f"loaded {len(health['loaded'])} · fallbacks {len(health['fallbacks'])}\n", style="#E8F4FF")
        tile.append(f"missing optional {len(health['missing_optional'])} · blocking {len(health['blocking_missing'])}", style="#B7D2E6")
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
            artifacts.append(line + "\n", style="#7CD66B" if line.startswith("✓") else "#A8C7DB")
        return artifacts

    def _issues(self) -> Text:
        return Text("none blocking", style="bold #7CD66B") if not self.inspection.blocking_issues else Text("\n".join(self.inspection.blocking_issues), style="bold #EF4444")

    def _next(self) -> Text:
        return _rows_text([("Technical", "Stage 09: LTX I2V takes"), ("Operator", "Improve CLI cockpit")], label_width=10)


def _row(label: str, value: str) -> str:
    return f"{label.upper():<16} {value}"


def _rows_text(rows: list[tuple[str, str]], *, label_width: int = 16) -> Text:
    text = Text()
    for label, value in rows:
        text.append(f"{label.upper():<{label_width}} ", style="bold #16C6FF")
        text.append(f"{value}\n", style=_value_style(value))
    return text


def _value_style(value: str) -> str:
    if value.startswith("✓"):
        return "bold #7CD66B"
    if value.startswith("?") or value.startswith("-"):
        return "#A8C7DB"
    if "Stage 09" in value or value == "paused":
        return "bold #FFB347"
    return "#E8F4FF"


def _status_text_style(status: str) -> str:
    if status == "passed":
        return "#7CD66B"
    if status == "pending":
        return "#A8C7DB"
    if status in {"needs_review", "unknown"}:
        return "#FFB347"
    if status == "rejected":
        return "#EF4444"
    return "#E8F4FF"


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


def run_cockpit(job_id: str, runs_root: str | Path) -> None:
    CreativeOSCockpitApp(CockpitArgs(job_id=job_id, runs_root=Path(runs_root))).run()
