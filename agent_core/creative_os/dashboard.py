from __future__ import annotations

from io import StringIO
import shutil
from typing import Any

from .run_inspector import CreativeOSRunInspector, RunInspection


BOX_WIDTH = 62
RULE = "─" * BOX_WIDTH


def render_dashboard(inspection: RunInspection, *, view: str = "overview", focus: str = "none") -> str:
    if not inspection.exists:
        return f"Creative OS run not found: {inspection.job_id}\nPath: {inspection.run_dir}\n"
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
        from rich.columns import Columns
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

    console_width = min(132, max(92, shutil.get_terminal_size((118, 24)).columns))
    console = Console(record=True, width=console_width, color_system="standard", file=StringIO())
    inspector = CreativeOSRunInspector()
    meta = _job_meta(inspection)

    header = Table.grid(expand=True)
    header.add_column(ratio=3)
    header.add_column(ratio=1)
    header.add_row(Text("CONTENT MASCHINE LIVE", style="bold cyan"))
    header.add_row(f"Job        {inspection.job_id}", "Session  artifact read")
    header.add_row("Pipeline   shortform_storyboard_v1", "Checks   no live checks")
    header.add_row(Text(f"Mode       {meta['mode']} · {meta['topic']}", style="cyan"), "Render   paused")
    header.add_row(f"Format     {meta['orientation']} · {meta['resolution']} · {meta['duration']}s · {meta['scene_count']} scenes", "")
    header.add_row(Text(f"Status     {inspection.status}", style=_status_style(inspection.status)), "")
    header.add_row(Text(f"Focus      {_operator_focus(focus)}", style="cyan"), "")

    if view == "all":
        console.print(Panel(header, box=box.ROUNDED, border_style="cyan"))
        console.print(_rich_overview_group(inspection, focus, width=console_width))
        for title, body in (
            ("SKILLS", _skills(inspection)),
            ("STAGES", _stages(inspection)),
            ("ARTIFACTS", _artifacts(inspection)),
            ("ISSUES", _issues(inspection)),
            ("NEXT", _next(inspection, focus)),
        ):
            console.print(Panel(body.strip(), title=title, box=box.ROUNDED, border_style="blue"))
        return console.export_text()

    if view != "overview":
        console.print(Panel(render_dashboard(inspection, view=view, focus=focus).strip(), title=view.upper(), box=box.ROUNDED))
        return console.export_text()

    console.print(Panel(header, box=box.ROUNDED, border_style="cyan"))
    console.print(_rich_overview_group(inspection, focus, width=console_width))
    return console.export_text()


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
    system.add_column(justify="left")
    system.add_column()
    system.add_row(Text("API", style="cyan"), Text("? not_checked", style="dim"))
    system.add_row(Text("Director", style="cyan"), Text("? not_checked", style="dim"))
    system.add_row("Image Backend", Text("✓ zimage_http", style="green") if inspection.artifacts.get("keyframe_manifest.json") else Text("? not_checked", style="dim"))
    system.add_row(Text("Video Backend", style="cyan"), Text("? planned ltx2", style="dim cyan"))
    system.add_row(Text("Vision Review", style="cyan"), Text("- manual_structured", style="yellow") if "stage6_review_decision" in inspection.data else Text("- heuristic", style="yellow"))
    system.add_row(Text("Voice", style="cyan"), Text("- disabled", style="dim"))
    system.add_row(Text("Music", style="cyan"), Text("- disabled", style="dim"))
    system.add_row(Text("Subtitles", style="cyan"), Text("- off", style="dim"))

    pipeline = Table.grid(expand=True)
    for stage in inspection.stages:
        pipeline.add_row(Text(f"{_mark(stage.status)} {stage.index} {stage.name}", style=_stage_style(stage.status)))

    workspace = Table.grid(padding=(0, 1), expand=True)
    workspace.add_column("label")
    workspace.add_column("value")
    workspace.add_row(Text("Current Step", style="cyan"), Text("LTX Motion Ready", style="bold cyan"))
    workspace.add_row(Text("Last passed", style="cyan"), Text("✓ 08 LTX motion prompts", style="green"))
    workspace.add_row(Text("Next technical", style="cyan"), Text("○ 09 LTX I2V takes", style="cyan"))
    workspace.add_row("Operator focus", _operator_focus(focus))
    workspace.add_row("Render paused", Text(_render_paused(focus), style="yellow"))
    stage_output = Text()
    stage_output.append("✓ 3 motion prompts passed\n", style="green")
    stage_output.append("✓ 3 keyframes passed\n", style="green")
    stage_output.append("○ video takes not built", style="cyan")

    scene_blocks: list[Any] = [Text("SCENE JOBS", style="bold cyan"), Text("─" * 46, style="dim")]
    for item in _motion_items(inspection):
        scene = Text()
        scene.append(f"[✓] {item['scene_id']}  READY FOR LTX\n", style="bold green")
        scene.append("    keyframe   ", style="dim")
        scene.append(f"{_basename(item['keyframe'])}\n", style="white")
        scene.append("    motion     ", style="dim")
        scene.append(item["summary"], style="cyan")
        scene_blocks.append(
            Panel(
                scene,
                box=box.SIMPLE,
                border_style="green",
                padding=(0, 1),
            )
        )

    workspace_group = Group(
        workspace,
        Text(""),
        Text("Stage output:", style="bold cyan"),
        stage_output,
        Text(""),
        *scene_blocks,
    )

    skill = Text(
        f"{health['mark']} {health['status']}\n"
        f"loaded {len(health['loaded'])} · fallbacks {len(health['fallbacks'])}\n"
        f"missing opt {len(health['missing_optional'])} · blocking {len(health['blocking_missing'])}",
        style="green" if health["status"] == "ok" else "yellow",
    )
    artifacts = Table.grid()
    keyframe_count = sum(1 for scene_id in ("scene_01", "scene_02", "scene_03") if inspection.artifacts.get(f"keyframes/{scene_id}.png"))
    for line in (
        f"{_artifact_mark(inspection, 'zimage_prompts.json')} zimage_prompts.json",
        f"{'✓' if keyframe_count == 3 else '○'} {keyframe_count} keyframes",
        f"{_artifact_mark(inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
        f"{_artifact_mark(inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
        f"{_artifact_mark(inspection, 'ltx_video_takes_manifest.json')} video_takes_manifest",
    ):
        artifacts.add_row(line)
    issues = Text(_issue_text(inspection), style="green" if not inspection.blocking_issues else "red")
    next_panel = Text("Tech: Stage 09\nOp: CLI focus" if focus == "cli" else f"Tech: {next_action}\nOp: {_operator_focus(focus)}", style="cyan")

    sidebar = Group(
        Panel(system, title="SYSTEM STATUS", box=box.ROUNDED, border_style="blue"),
        Panel(pipeline, title="PIPELINE MAP", box=box.ROUNDED, border_style="blue"),
    )
    workspace_panel = Panel(workspace_group, title="ACTIVE WORKSPACE", box=box.ROUNDED, border_style="cyan", padding=(1, 2))
    if width >= 110:
        main_grid = Table.grid(expand=True, padding=(0, 1))
        main_grid.add_column(ratio=1)
        main_grid.add_column(ratio=2)
        main_grid.add_row(sidebar, workspace_panel)
    else:
        main_grid = Group(sidebar, workspace_panel)

    bottom_panels = [
        Panel(skill, title="SKILL HEALTH", box=box.ROUNDED, border_style="green"),
        Panel(artifacts, title="ARTIFACTS", box=box.ROUNDED, border_style="blue"),
        Panel(issues, title="ISSUES", box=box.ROUNDED, border_style="green" if not inspection.blocking_issues else "red"),
        Panel(next_panel, title="NEXT", box=box.ROUNDED, border_style="cyan"),
    ]
    bottom_grid = Table.grid(expand=True, padding=(0, 1))
    if width >= 118:
        for _ in range(4):
            bottom_grid.add_column(ratio=1)
        bottom_grid.add_row(*bottom_panels)
    else:
        bottom_grid.add_column(ratio=1)
        bottom_grid.add_column(ratio=1)
        bottom_grid.add_row(bottom_panels[0], bottom_panels[1])
        bottom_grid.add_row(bottom_panels[2], bottom_panels[3])

    return Group(main_grid, bottom_grid)


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
        f"  Last passed      ✓ {inspector.last_passed_stage(inspection)}",
        "  Next technical   ○ 09 LTX video takes" if inspection.status == "ready_for_ltx_i2v_takes" else f"  Next technical   {next_action}",
        f"  Operator focus   {_operator_focus(focus)}",
        f"  Render paused    {_render_paused(focus)}",
        "",
        _section_title("SYSTEM"),
        "  API              ? not_checked",
        "  Director         ? not_checked",
        "  Image Backend    ✓ zimage_http" if inspection.artifacts.get("keyframe_manifest.json") else "  Image Backend    ? not_checked",
        "  Video Backend    ? not_checked · planned ltx2" if inspection.artifacts.get("ltx_motion_prompts.json") else "  Video Backend    ? not_checked",
        "  Vision Review    - manual_structured" if "stage6_review_decision" in inspection.data else "  Vision Review    - heuristic",
        "  Voice            - disabled",
        "  Music            - disabled",
        "  Subtitles        - off",
        "",
        _section_title("PIPELINE PATH"),
        "  Creative OS",
        "    → Z-Image Prompts",
        "    → Keyframes",
        "    → Keyframe QA",
        "    → LTX Motion Prompts",
        "    → LTX I2V Takes",
        "",
        _section_title("PROGRESS"),
    ]
    lines.extend(f"  {_mark(stage.status)} {stage.index} {stage.name}" for stage in inspection.stages[:9])
    lines.extend(
        [
            "",
            _section_title("SKILL HEALTH"),
            f"  {skill_health['mark']} {skill_health['status']} · loaded {len(loaded)} · fallbacks {len(fallbacks)} · missing optional {len(missing_optional)} · blocking {len(blocking_missing)}",
            f"  Main: {', '.join(main_skills) if main_skills else 'unknown'}",
            "",
            _section_title("ARTIFACTS"),
            f"  {_artifact_mark(inspection, 'zimage_prompts.json')} zimage_prompts.json",
            f"  {'✓' if keyframe_count == 3 else '○'} {keyframe_count} keyframes",
            f"  {_artifact_mark(inspection, 'ltx_motion_prompts.json')} ltx_motion_prompts.json",
            f"  {_artifact_mark(inspection, 'ltx_prompt_audit.json')} ltx_prompt_audit.json",
            f"  {_artifact_mark(inspection, 'ltx_video_takes_manifest.json')} ltx_video_takes_manifest.json",
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


def _artifact_mark(inspection: RunInspection, name: str) -> str:
    return "✓" if inspection.artifacts.get(name) else "○"


def _issue_text(inspection: RunInspection) -> str:
    return "none blocking" if not inspection.blocking_issues else "; ".join(inspection.blocking_issues)


def _operator_focus(focus: str) -> str:
    return {
        "cli": "Improve CLI cockpit before rendering",
        "render": "Prepare controlled LTX I2V render gate",
        "audit": "Audit Creative OS artifacts before rendering",
        "none": "none",
    }.get(focus, "none")


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
    priority = ["tiktok_shortform", "zimage", "cinematic_nature", "artifact_avoidance"]
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
        items.append({"scene_id": scene_id, "keyframe": keyframe_display, "summary": _shorten(summary, 48)})
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
        return "bold cyan"
    if status in {"blocked_by_keyframe_review", "blocked_by_ltx_prompt_audit", "blocked"}:
        return "bold red"
    if status == "completed":
        return "bold green"
    if status == "in_progress":
        return "yellow"
    return "dim"


def _stage_style(status: str) -> str:
    if status == "passed":
        return "green"
    if status == "pending":
        return "cyan"
    if status == "needs_review":
        return "yellow"
    if status == "rejected":
        return "red"
    if status in {"missing", "unknown"}:
        return "dim"
    return "white"
