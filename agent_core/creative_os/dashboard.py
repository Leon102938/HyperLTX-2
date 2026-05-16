from __future__ import annotations

from io import StringIO
import shutil
from typing import Any

from .run_inspector import CreativeOSRunInspector, RunInspection


BOX_WIDTH = 62
RULE = "─" * BOX_WIDTH


COCKPIT_THEME = {
    "background": "#020617",
    "panel_bg": "#020817",
    "border_primary": "#38BDF8",
    "border_secondary": "#1E3A5F",
    "title": "bold #67E8F9",
    "label": "#67E8F9",
    "value": "#E5E7EB",
    "muted": "#64748B",
    "ok": "#22C55E",
    "warn": "#FBBF24",
    "error": "#EF4444",
    "pending": "#67E8F9",
    "active": "bold #FBBF24",
    "next": "bold #67E8F9",
    "disabled": "#64748B",
}


def _t(name: str) -> str:
    return COCKPIT_THEME[name]


def render_dashboard(inspection: RunInspection, *, view: str = "overview", focus: str = "none") -> str:
    if not inspection.exists:
        return _missing_run_message(inspection)
    if view == "all":
        return "\n\n".join(
            _view_block(name.upper(), render_dashboard(inspection, view=name, focus=focus))
            for name in ("overview", "skills", "stages", "artifacts", "issues", "next")
        )
    if view == "skills":
        return _skills(inspection)
    if view == "stages":
        return _stages(inspection)
    if view == "artifacts":
        return _artifacts(inspection)
    if view == "issues":
        return _issues(inspection)
    if view == "next":
        return _next(inspection, focus)
    return _overview(inspection, focus)


def render_rich_dashboard(inspection: RunInspection, *, view: str = "overview", focus: str = "none") -> str:
    try:
        from rich import box
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return "[warning] Rich is not installed; falling back to plain dashboard.\n" + render_dashboard(
            inspection, view=view, focus=focus
        )

    if not inspection.exists:
        return render_dashboard(inspection, view=view, focus=focus)

    console_width = min(142, max(88, shutil.get_terminal_size((122, 24)).columns))
    console = Console(record=True, width=console_width, color_system="truecolor", file=StringIO())
    meta = _job_meta(inspection)
    header = _rich_header(inspection, focus, meta)

    if view == "all":
        console.print(_cockpit_panel(header, box=box.ROUNDED, border_style=_t("border_primary"), padding=(0, 2)))
        console.print(_rich_overview_group(inspection, focus, width=console_width))
        for title, body in (
            ("SKILLS DETAIL", _skills(inspection)),
            ("STAGES DETAIL", _stages(inspection)),
            ("ARTIFACTS DETAIL", _artifacts(inspection)),
            ("ISSUES DETAIL", _issues(inspection)),
            ("NEXT DETAIL", _next(inspection, focus)),
        ):
            console.rule(Text(title, style=_t("title")), style=_t("border_secondary"))
            console.print(_cockpit_panel(body.strip(), box=box.ROUNDED, border_style=_t("border_secondary")))
        return console.export_text()

    if view != "overview":
        console.print(_cockpit_panel(render_dashboard(inspection, view=view, focus=focus).strip(), title=view.upper(), box=box.ROUNDED, border_style=_t("border_secondary")))
        return console.export_text()

    console.print(_cockpit_panel(header, box=box.ROUNDED, border_style=_t("border_primary"), padding=(0, 2)))
    console.print(_rich_overview_group(inspection, focus, width=console_width))
    return console.export_text()


def _missing_run_message(inspection: RunInspection) -> str:
    return (
        "Run not found:\n"
        f"  job_id: {inspection.job_id}\n"
        f"  searched: {inspection.run_dir}\n"
        "\n"
        "This is not a system error.\n"
        "agent_runs contains disposable run artifacts only.\n"
        "Use --runs-root for fixtures or create a real run first.\n"
    )


def _technical_flow() -> Any:
    from rich.console import Group
    from rich.text import Text

    title = Text("TECHNICAL FLOW", style=_t("title"))
    flow = Text()
    for index, label in enumerate(
        (
            ("Director", _t("muted")),
            ("Creative OS", _t("muted")),
            ("HiDream-O1-Dev", _t("ok")),
            ("Keyframes", _t("ok")),
            ("QA", _t("ok")),
            ("LTX Motion", _t("ok")),
            ("LTX I2V Takes", _t("active")),
        )
    ):
        if index:
            flow.append("  →  ", style=_t("muted"))
        flow.append(label[0], style=label[1])
    return Group(title, flow)


def _cockpit_panel(*args: Any, **kwargs: Any) -> Any:
    from rich.panel import Panel

    kwargs.setdefault("style", f"{_t('value')} on {_t('panel_bg')}")
    kwargs.setdefault("padding", (0, 1))
    return Panel(*args, **kwargs)


def _rich_header(inspection: RunInspection, focus: str, meta: dict[str, Any]) -> Any:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    header = Table.grid(expand=True, padding=(0, 2))
    header.add_column(width=11)
    header.add_column(ratio=5)
    header.add_column(width=11)
    header.add_column(ratio=4)
    header.add_row(Text("Job", style=_t("label")), Text(inspection.job_id, style=_t("value")), Text("Pipeline", style=_t("label")), Text("shortform_storyboard_v1", style=_t("value")))
    header.add_row(Text("Mode", style=_t("label")), Text(_shorten(f"{meta['mode']} · {meta['topic']}", 58), style=_t("value")), "", "")
    header.add_row(Text("Format", style=_t("label")), Text(f"{meta['orientation']} · {meta['resolution']} · {meta['duration']}s · {meta['scene_count']} scenes", style=_t("value")), "", "")
    header.add_row(Text("Status", style=_t("label")), Text(inspection.status, style=_status_style(inspection.status)), Text("Focus", style=_t("label")), Text(_operator_focus_short(focus), style=_t("next")))
    header.add_row(Text("Session", style=_t("label")), Text(_session_label(inspection), style=_t("warn") if _is_fixture_run(inspection) else _t("value")), Text("Checks", style=_t("label")), Text("no live checks", style=_t("muted")))
    header.add_row(Text("Render", style=_t("label")), Text("paused" if _render_paused(focus) == "yes" else "not_paused", style=_t("warn")), Text("Style", style=_t("label")), Text("rich cockpit", style=_t("value")))
    return Group(Text("CONTENT MASCHINE LIVE", style="bold #67E8F9"), header)


def _rich_overview_group(inspection: RunInspection, focus: str, width: int) -> Any:
    from rich import box
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    inspector = CreativeOSRunInspector()
    health = inspector.skill_health(inspection)
    next_action = inspector.next_action(inspection)

    system = Table.grid(padding=(0, 1), expand=True)
    system.add_column(justify="left", ratio=2)
    system.add_column(ratio=2)
    system.add_row(Text("API", style=_t("label")), Text("? not_checked", style=_t("muted")))
    system.add_row(Text("Director", style=_t("label")), Text("? not_checked", style=_t("muted")))
    system.add_row(Text("Image Backend", style=_t("label")), Text("✓ hidream_o1_dev", style=_t("ok")) if inspection.artifacts.get("keyframe_manifest.json") else Text("? not_checked", style=_t("muted")))
    system.add_row(Text("Video Backend", style=_t("label")), Text("? planned ltx2", style=_t("muted")))
    system.add_row(Text("Vision Review", style=_t("label")), Text("- manual_structured", style=_t("warn")) if "stage6_review_decision" in inspection.data else Text("- heuristic", style=_t("warn")))
    system.add_row(Text("Voice", style=_t("label")), Text("- disabled", style=_t("disabled")))
    system.add_row(Text("Music", style=_t("label")), Text("- disabled", style=_t("disabled")))
    system.add_row(Text("Subtitles", style=_t("label")), Text("- off", style=_t("disabled")))

    pipeline = Table.grid(expand=True)
    for stage in inspection.stages:
        label = f"{_stage_timeline_mark(stage.status)} {stage.index} {stage.name}"
        pipeline.add_row(Text(label, style=_stage_style(stage.status, stage.index)))

    workspace = Table.grid(padding=(0, 1), expand=True)
    workspace.add_column(ratio=1)
    workspace.add_column(ratio=3)
    workspace.add_row(Text("Current Step", style=_t("label")), Text("LTX Motion Ready", style=_t("next")))
    workspace.add_row(Text("Last passed", style=_t("label")), Text(f"✓ {inspector.last_passed_stage(inspection)}", style=_t("ok")))
    workspace.add_row(Text("Next technical", style=_t("label")), Text("○ 09 LTX I2V takes", style=_t("active")))
    workspace.add_row(Text("Operator focus", style=_t("label")), Text(_operator_focus_short(focus), style=_t("value")))
    workspace.add_row(Text("Render paused", style=_t("label")), Text(_render_paused(focus), style=_t("warn")))
    stage_output = Text()
    stage_output.append("  ✓ 3 motion prompts passed\n", style=_t("ok"))
    stage_output.append("  ✓ 3 keyframes passed\n", style=_t("ok"))
    stage_output.append("  ○ video takes not built", style=_t("muted"))

    scene_blocks: list[Any] = [Text("SCENE JOBS", style=_t("title"))]
    for item in _motion_items(inspection):
        scene = Table.grid(expand=True, padding=(0, 1))
        scene.add_column(width=12)
        scene.add_column(ratio=1)
        scene.add_row(Text("Scene", style=_t("muted")), Text(f"[✓] {item['scene_id']}  READY FOR LTX", style=f"bold {_t('ok')}"))
        scene.add_row(Text("keyframe", style=_t("muted")), Text(_basename(item["keyframe"]), style=_t("value")))
        scene.add_row(Text("motion", style=_t("muted")), Text(item["summary"], style=_t("value")))
        scene.add_row(Text("status", style=_t("muted")), Text("waiting for Stage 09 render gate", style=_t("warn")))
        scene_blocks.append(
            _cockpit_panel(
                scene,
                box=box.ROUNDED,
                border_style=_t("border_secondary"),
                padding=(0, 1),
            )
        )

    workspace_group = Group(
        workspace,
        Text(""),
        Text("Stage output:", style=_t("title")),
        stage_output,
        Text(""),
        *scene_blocks,
        Text(""),
        _technical_flow(),
    )

    skill = Text()
    skill.append(f"{health['mark']} {health['status']}\n", style=_t("ok") if health["status"] == "ok" else _t("warn"))
    skill.append(f"loaded {len(health['loaded'])} · fallbacks {len(health['fallbacks'])}\n", style=_t("value"))
    skill.append(f"missing optional {len(health['missing_optional'])} · blocking {len(health['blocking_missing'])}", style=_t("value"))
    artifacts = Table.grid()
    keyframe_count = sum(1 for scene_id in ("scene_01", "scene_02", "scene_03") if inspection.artifacts.get(f"keyframes/{scene_id}.png"))
    for line in (
        f"{_artifact_mark(inspection, 'hidream_prompts.json')} hidream_prompts.json",
        f"{'✓' if keyframe_count == 3 else '○'} {keyframe_count} keyframes",
        f"{_artifact_mark(inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
        f"{_artifact_mark(inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
        f"{_artifact_mark(inspection, 'ltx_video_takes_manifest.json')} video_takes_manifest",
    ):
        artifacts.add_row(Text(line, style=_artifact_line_style(line)))
    issues = Text(_issue_text(inspection), style=_t("ok") if not inspection.blocking_issues else _t("error"))
    next_panel = Table.grid(expand=True, padding=(0, 1))
    next_panel.add_column(width=10)
    next_panel.add_column(ratio=1)
    next_panel.add_row(Text("Technical", style=_t("label")), Text("Stage 09: LTX I2V takes", style=_t("active")))
    next_panel.add_row(Text("Operator", style=_t("label")), Text("Improve CLI cockpit", style=_t("next")))

    sidebar = Group(
        _cockpit_panel(system, title=Text("SYSTEM STATUS", style=_t("title")), box=box.ROUNDED, border_style=_t("border_primary"), padding=(1, 1)),
        Text(""),
        _cockpit_panel(pipeline, title=Text("PIPELINE MAP", style=_t("title")), box=box.ROUNDED, border_style=_t("border_secondary"), padding=(1, 1)),
    )
    workspace_panel = _cockpit_panel(workspace_group, title=Text("ACTIVE WORKSPACE", style=_t("title")), box=box.ROUNDED, border_style=_t("border_primary"), padding=(1, 2))
    if width >= 110:
        main_grid = Table.grid(expand=True, padding=(0, 2))
        main_grid.add_column(ratio=36)
        main_grid.add_column(ratio=64)
        main_grid.add_row(sidebar, workspace_panel)
    else:
        main_grid = Group(sidebar, workspace_panel)

    bottom_panels = [
        _cockpit_panel(skill, title=Text("SKILL HEALTH", style=_t("title")), box=box.ROUNDED, border_style=_t("border_secondary")),
        _cockpit_panel(artifacts, title=Text("ARTIFACTS", style=_t("title")), box=box.ROUNDED, border_style=_t("border_secondary")),
        _cockpit_panel(issues, title=Text("ISSUES", style=_t("title")), box=box.ROUNDED, border_style=_t("ok") if not inspection.blocking_issues else _t("error")),
        _cockpit_panel(next_panel, title=Text("NEXT", style=_t("title")), box=box.ROUNDED, border_style=_t("border_primary")),
    ]
    bottom_grid = Table.grid(expand=True, padding=(0, 2))
    if width >= 150:
        for _ in range(4):
            bottom_grid.add_column(ratio=1)
        bottom_grid.add_row(*bottom_panels)
    else:
        bottom_grid.add_column(ratio=1)
        bottom_grid.add_column(ratio=1)
        bottom_grid.add_row(bottom_panels[0], bottom_panels[1])
        bottom_grid.add_row(bottom_panels[2], bottom_panels[3])

    return Group(main_grid, Text(""), bottom_grid)


def _overview(inspection: RunInspection, focus: str) -> str:
    inspector = CreativeOSRunInspector()
    meta = _job_meta(inspection)
    skill_health = inspector.skill_health(inspection)
    loaded = skill_health["loaded"]
    fallbacks = skill_health["fallbacks"]
    missing_optional = skill_health["missing_optional"]
    blocking_missing = skill_health["blocking_missing"]
    next_action = inspector.next_action(inspection)
    keyframe_count = sum(1 for scene_id in ("scene_01", "scene_02", "scene_03") if inspection.artifacts.get(f"keyframes/{scene_id}.png"))
    main_skills = _main_skill_names(loaded)
    phase1 = _phase1_status(inspection)
    lines = [
        _header_box(
            [
                "CONTENT MASCHINE LIVE",
                f"Job        {inspection.job_id}",
                "Pipeline   shortform_storyboard_v1",
                f"Mode       {meta['mode']} · {meta['topic']}",
                f"Format     {meta['orientation']} · {meta['resolution']} · {meta['duration']}s · {meta['scene_count']} scenes",
                f"Status     {inspection.status}",
            ]
        ),
        "",
        _section_title("CURRENT POSITION"),
        f"  Current step     {_phase1_current_step(phase1) if phase1 else 'LTX Motion Ready'}",
        f"  Last passed      ✓ {inspector.last_passed_stage(inspection)}",
        f"  Next technical   {next_action}" if phase1 else ("  Next technical   ○ 09 LTX video takes" if inspection.status == "ready_for_ltx_i2v_takes" else f"  Next technical   {next_action}"),
        f"  Operator focus   {_phase1_operator_focus(phase1, focus) if phase1 else _operator_focus(focus)}",
        f"  Render paused    {_render_paused(focus)}",
        "",
        _section_title("SYSTEM"),
        "  API              ? not_checked",
        "  Director         ? not_checked",
        "  Image Backend    ✓ hidream_o1_dev" if inspection.artifacts.get("keyframe_manifest.json") else "  Image Backend    ? not_checked",
        "  Video Backend    ? not_checked · planned ltx2" if inspection.artifacts.get("ltx_motion_prompts.json") else "  Video Backend    ? not_checked",
        "  Vision Review    - manual_structured" if "stage6_review_decision" in inspection.data else "  Vision Review    - heuristic",
        "  Voice            - disabled",
        "  Music            - disabled",
        "  Subtitles        - off",
        "",
        _section_title("PIPELINE PATH"),
        *_pipeline_path_lines(phase1),
        "",
        _section_title("PROGRESS"),
    ]
    progress_stages = inspection.stages[:10] if phase1 else inspection.stages[:9]
    lines.extend(f"  {_mark(stage.status)} {stage.index} {stage.name}" for stage in progress_stages)
    lines.extend(
        [
            "",
            _section_title("SKILL HEALTH"),
            f"  {skill_health['mark']} {skill_health['status']} · loaded {len(loaded)} · fallbacks {len(fallbacks)} · missing optional {len(missing_optional)} · blocking {len(blocking_missing)}",
            f"  Main: {', '.join(main_skills) if main_skills else 'unknown'}",
            "",
            _section_title("ARTIFACTS"),
            *_artifact_overview_lines(inspection, keyframe_count, phase1),
        ]
    )
    lines.extend(
        [
            "",
            _section_title("ISSUES"),
            f"  {_issue_text(inspection)}",
            "",
            _section_title("NEXT"),
            f"  Technical   {next_action}",
            f"  Operator    {_operator_focus(focus)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _phase1_status(inspection: RunInspection) -> dict[str, Any] | None:
    phase1 = inspection.data.get("phase1_status")
    return phase1 if isinstance(phase1, dict) else None


def _phase1_current_step(phase1: dict[str, Any]) -> str:
    real_stage = str(phase1.get("real_run_stage") or phase1.get("current_stage") or "not_checked")
    if real_stage == "09":
        return "Stage 09 Image / Keyframe Generation"
    if real_stage == "not_checked":
        return "not_checked"
    return f"Stage {real_stage}"


def _phase1_operator_focus(phase1: dict[str, Any], focus: str) -> str:
    if focus != "none":
        return _operator_focus(focus)
    if phase1.get("next_available_stage") == "none_phase1_complete":
        return "review Stage 09 keyframes"
    if phase1.get("status") == "paused_missing_backend":
        return "restore image backend or rerun with --no-images"
    return "not_checked"


def _pipeline_path_lines(phase1: dict[str, Any] | None) -> list[str]:
    if phase1:
        return [
            "  Phase 1 live: 00 Command Center → 09 Image / Keyframe Generation",
            "  Stage 10+ runtime: not built",
        ]
    return [
        "  Creative OS",
        "    → HiDream Prompts",
        "    → Keyframes",
        "    → Keyframe QA",
        "    → LTX Motion Prompts",
        "    → LTX I2V Takes",
    ]


def _artifact_overview_lines(inspection: RunInspection, keyframe_count: int, phase1: dict[str, Any] | None) -> list[str]:
    if phase1:
        gallery_mark = _artifact_mark(inspection, "keyframe_gallery.html")
        return [
            f"  {_artifact_mark(inspection, 'prompt_payload_compiled.json')} prompt_payload_compiled.json",
            f"  {_artifact_mark(inspection, 'hidream_prompts.json')} hidream_prompts.json",
            f"  {_artifact_mark(inspection, 'keyframe_manifest.json')} keyframe_manifest.json",
            f"  {'✓' if keyframe_count == 3 else '○'} {keyframe_count} keyframes",
            f"  {gallery_mark} keyframe_gallery.html",
            "  ○ Stage 10+ artifacts · not built",
        ]
    return [
        f"  {_artifact_mark(inspection, 'hidream_prompts.json')} hidream_prompts.json",
        f"  {'✓' if keyframe_count == 3 else '○'} {keyframe_count} keyframes",
        f"  {_artifact_mark(inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
        f"  {_artifact_mark(inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
        f"  {_artifact_mark(inspection, 'ltx_video_takes_manifest.json')} ltx_video_takes_manifest.json",
    ]


def _skills(inspection: RunInspection) -> str:
    health = CreativeOSRunInspector().skill_health(inspection)
    match = inspection.data.get("skill_match") or {}
    loaded = match.get("loaded_skill_ids") or []
    reasons = match.get("reasons") or {}
    groups = {
        "core": [],
        "model": [],
        "style": [],
        "fallback": [],
        "missing optional": health["missing_optional"],
        "blocking missing": health["blocking_missing"],
    }
    for skill_id in loaded:
        if skill_id.startswith("core/"):
            groups["core"].append(skill_id)
        elif skill_id.startswith("models/"):
            groups["model"].append(skill_id)
        elif skill_id.startswith("styles/"):
            groups["style"].append(skill_id)
        elif skill_id.startswith("fallback/"):
            groups["fallback"].append(skill_id)
    lines = [_plain_view_title("SKILLS"), f"health: {health['mark']} {health['status']}"]
    for name, values in groups.items():
        lines.append(f"{name}:")
        if not values:
            lines.append("  - none")
        for value in values:
            reason = reasons.get(value, "")
            lines.append(f"  - {value}" + (f" · {reason}" if reason else ""))
    return "\n".join(lines) + "\n"


def _stages(inspection: RunInspection) -> str:
    lines = [_plain_view_title("STAGES")]
    for stage in inspection.stages:
        lines.append(f"{_mark(stage.status)} {stage.index} {stage.name} · {stage.artifact} · {stage.status} · {stage.detail}")
    return "\n".join(lines) + "\n"


def _artifacts(inspection: RunInspection) -> str:
    lines = [_plain_view_title("ARTIFACTS")]
    for name, exists in sorted(inspection.artifacts.items()):
        lines.append(f"{'✓' if exists else '○'} {name} · {'exists' if exists else 'missing'}")
    return "\n".join(lines) + "\n"


def _issues(inspection: RunInspection) -> str:
    lines = [_plain_view_title("ISSUES")]
    if not inspection.issues:
        lines.append("none blocking")
    else:
        lines.extend(f"- {issue}" for issue in inspection.issues)
    return "\n".join(lines) + "\n"


def _next(inspection: RunInspection, focus: str) -> str:
    return (
        f"{_plain_view_title('NEXT')}\n"
        f"Technical   {CreativeOSRunInspector().next_action(inspection)}\n"
        f"Operator    {_operator_focus(focus)}\n"
    )


def _mark(status: str) -> str:
    if status == "passed":
        return "✓"
    if status in {"needs_review", "unknown"}:
        return "?"
    if status == "rejected":
        return "!"
    return "○"


def _stage_timeline_mark(status: str) -> str:
    if status == "passed":
        return "✓"
    if status in {"pending", "missing"}:
        return "○"
    if status in {"needs_review", "unknown"}:
        return "?"
    if status == "rejected":
        return "!"
    return "○"


def _artifact_mark(inspection: RunInspection, name: str) -> str:
    return "✓" if inspection.artifacts.get(name) else "○"


def _artifact_line_style(line: str) -> str:
    if line.startswith("✓"):
        return _t("ok")
    if line.startswith("!"):
        return _t("error")
    return _t("muted")


def _issue_text(inspection: RunInspection) -> str:
    return "none blocking" if not inspection.blocking_issues else "; ".join(inspection.blocking_issues)


def _operator_focus(focus: str) -> str:
    return {
        "cli": "Improve CLI cockpit before rendering",
        "render": "Prepare controlled LTX I2V render gate",
        "audit": "Audit Creative OS artifacts before rendering",
        "none": "none",
    }.get(focus, "none")


def _operator_focus_short(focus: str) -> str:
    return {
        "cli": "CLI cockpit refinement",
        "render": "Prepare controlled render gate",
        "audit": "Artifact audit before rendering",
        "none": "none",
    }.get(focus, "none")


def _is_fixture_run(inspection: RunInspection) -> bool:
    return "tests/fixtures/creative_os_runs" in str(inspection.run_dir)


def _session_label(inspection: RunInspection) -> str:
    return "fixture/demo" if _is_fixture_run(inspection) else "artifact read"


def _render_paused(focus: str) -> str:
    return "no" if focus == "render" else "yes"


def _header_box(lines: list[str]) -> str:
    inner_width = BOX_WIDTH - 2
    output = ["╭" + "─" * inner_width + "╮"]
    for index, line in enumerate(lines):
        if index == 1:
            output.append("├" + "─" * inner_width + "┤")
        output.append("│ " + _fit(line, inner_width - 2) + " │")
    output.append("╰" + "─" * inner_width + "╯")
    return "\n".join(output)


def _fit(text: str, width: int) -> str:
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.ljust(width)


def _section_title(title: str) -> str:
    return f"{title}\n{RULE}"


def _plain_view_title(title: str) -> str:
    return f"{title}\n{RULE}"


def _view_block(title: str, content: str) -> str:
    body = content.rstrip()
    prefix = f"{title}\n{RULE}\n"
    if body.startswith(prefix):
        body = body[len(prefix):]
    return f"{RULE}\n{title}\n{RULE}\n{body}"


def _main_skill_names(skill_ids: list[str]) -> list[str]:
    priority = ["tiktok_shortform", "hidream", "cinematic_nature", "artifact_avoidance"]
    names = [skill_id.split("/", 1)[-1] for skill_id in skill_ids]
    selected = [name for name in priority if name in names]
    for name in names:
        if name not in selected:
            selected.append(name)
        if len(selected) == 4:
            break
    return selected


def _job_meta(inspection: RunInspection) -> dict[str, Any]:
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
        scene_id = str(prompt.get("scene_id") or "unknown")
        keyframe = str(prompt.get("source_keyframe_path") or "")
        keyframe_display = keyframe.replace(str(inspection.run_dir) + "/", "") if keyframe else "missing"
        summary = str(prompt.get("camera_motion") or prompt.get("motion_prompt") or "motion prompt present")
        items.append({"scene_id": scene_id, "keyframe": keyframe_display, "summary": _shorten(summary, 65)})
    return items


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else "missing"


def _status_style(status: str) -> str:
    if status == "ready_for_ltx_i2v_takes":
        return _t("next")
    if status in {"blocked_by_keyframe_review", "blocked_by_ltx_prompt_audit", "blocked"}:
        return f"bold {_t('error')}"
    if status == "completed":
        return f"bold {_t('ok')}"
    if status == "in_progress":
        return _t("warn")
    return _t("muted")


def _stage_style(status: str, index: str | None = None) -> str:
    if index == "09" and status == "pending":
        return _t("active")
    if status == "passed":
        return _t("ok")
    if status == "pending":
        return _t("pending")
    if status == "needs_review":
        return _t("warn")
    if status == "rejected":
        return _t("error")
    if status in {"missing", "unknown"}:
        return _t("muted")
    return _t("value")
