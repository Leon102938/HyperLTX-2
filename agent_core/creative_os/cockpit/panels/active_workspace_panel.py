from __future__ import annotations

from rich.text import Text

from agent_core.creative_os.cockpit.panels.common import rows_text
from agent_core.creative_os.cockpit.state_adapter import CockpitState
from agent_core.creative_os.cockpit.theme import BG_WORKSPACE, TEXT_LABEL, TEXT_MAIN, TEXT_MUTED, TEXT_SUCCESS, style

BOX_WIDTH = 132
JOB_PREVIEW_WIDTH = 18
JOB_MAIN_WIDTH = 78
JOB_STATUS_WIDTH = 20


def render(state: CockpitState) -> Text:
    if not state.run_found:
        return _missing_run_text(state)

    data = state.workspace
    workspace = Text()
    workspace.append(
        _section_box(
            "CURRENT POSITION",
            _position_grid(
                (
                    ("Current Step", data.current_step),
                    ("Operator Focus", data.operator_focus),
                    ("Render Paused", data.render_paused),
                    ("Final MP4", data.final_mp4 or "unknown"),
                    ("Last Passed", data.last_passed),
                    ("Next Technical", data.next_technical),
                    ("Director Mode", data.director_mode or "unknown"),
                    ("Run Type", data.run_type or state.run_type or "unknown"),
                )
            ),
        )
    )
    job_lines = _prompt_job_lines(state)
    workspace.append(_section_box("PROMPTS / IMAGE JOBS", job_lines))
    return workspace


def _prompt_job_lines(state: CockpitState) -> list[Text]:
    data = state.workspace
    if not data.scenes:
        line = Text()
        line.append("○ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        line.append("No prompt/image jobs available", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return [line]
    lines: list[Text] = []
    for index, scene in enumerate(data.scenes[:3], start=1):
        if index > 1:
            lines.append(_job_gap_line())
        lines.extend(_render_prompt_job(index, scene, expanded=index == 1))
    return lines


def _render_prompt_job(index: int, scene: object, *, expanded: bool) -> list[Text]:
    scene_id = _scene_value(scene, "scene_id", f"scene_{index:02d}")
    source = _scene_value(scene, "keyframe", "missing")
    summary = _scene_value(scene, "summary", "unknown")
    title = _scene_value(scene, "title", scene_id)
    state_label = _scene_value(scene, "state_label", "ready")
    output_status = _scene_value(scene, "status", "unknown")
    status = _status_for_job(state_label, output_status)
    caret = "v" if expanded else ">"

    lines = [_job_label_row(), _job_summary_row(index, scene_id, title, summary, source, status, caret, state_label, output_status)]
    if expanded:
        lines.extend(_job_detail_lines(scene, summary, source, status))
    return lines


def _job_label_row() -> Text:
    line = Text()
    line.append(f"{'PREVIEW':<{JOB_PREVIEW_WIDTH}}", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(f"{'JOB / PROMPT':<{JOB_MAIN_WIDTH}}", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(f"{'STATUS':<{JOB_STATUS_WIDTH}}", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    return line


def _job_summary_row(
    index: int,
    scene_id: str,
    title: str,
    summary: str,
    source: str,
    status: str,
    caret: str,
    state_label: str,
    output_status: str,
) -> Text:
    line = Text()
    line.append(f"{_shorten(_preview_label(source), JOB_PREVIEW_WIDTH):<{JOB_PREVIEW_WIDTH}}", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(f"{index:02d} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    line.append(f"{_shorten(scene_id, 12):<12}", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True))
    line.append(" ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    main_text = f"{title} · {summary}" if title != scene_id else summary
    line.append(f"{_shorten(main_text, JOB_MAIN_WIDTH - 17):<{JOB_MAIN_WIDTH - 17}}", style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    line.append(f"{_shorten(status, JOB_STATUS_WIDTH - 3):<{JOB_STATUS_WIDTH - 3}}", style=_job_status_style(state_label, output_status))
    line.append(caret, style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    return line


def _job_detail_lines(scene: object, summary: str, source: str, status: str) -> list[Text]:
    detail_fields = [
        ("prompt", _scene_optional(scene, "prompt") or summary),
        ("status", status),
        ("source", _basename(source)),
    ]
    for label in ("backend", "generator", "model", "seed", "output_path", "review_status", "queue_state", "elapsed"):
        value = _scene_optional(scene, label)
        if value:
            detail_fields.append((label, value))
    lines = [_detail_line(detail_fields[:3])]
    if len(detail_fields) > 3:
        lines.append(_detail_line(detail_fields[3:]))
    progress = _scene_optional(scene, "progress_percent")
    if progress:
        lines.append(_progress_line(progress))
    return lines


def _detail_line(fields: list[tuple[str, str]]) -> Text:
    line = Text()
    line.append("details  ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    for index, (label, value) in enumerate(fields):
        if index:
            line.append("  |  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        line.append(f"{label}: ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        line.append(_shorten(value, 34), style=_value_style(value))
    return line


def _progress_line(progress: str) -> Text:
    line = Text()
    line.append("progress ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    try:
        percent = max(0, min(100, int(float(progress))))
    except ValueError:
        line.append(_shorten(progress, 80), style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return line
    filled = percent // 10
    line.append("[" + "#" * filled + "-" * (10 - filled) + f"] {percent}%", style=style(TEXT_SUCCESS, bg=BG_WORKSPACE))
    return line


def _job_gap_line() -> Text:
    line = Text()
    line.append("·" * 126, style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return line


def _preview_label(source: str) -> str:
    basename = _basename(source)
    if basename in {"missing", "unknown", "not_checked"}:
        return "empty preview"
    return basename


def _missing_run_text(state: CockpitState) -> Text:
    workspace = rows_text(
        [
            ("Current Step", "Run not found"),
            ("Searched", str(state.data_source_path)),
            ("Hint", "use --runs-root for fixture/demo data or create a real run first"),
            ("Watch", "on" if state.watch_enabled else "off"),
            ("Render paused", "unknown"),
            ("Run type", "missing"),
            ("Final MP4", "unknown"),
            ("Director Mode", "unknown"),
        ],
        label_width=17,
        bg=BG_WORKSPACE,
    )
    workspace.append("\nRun not found\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    workspace.append(f"searched: {state.data_source_path}\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    workspace.append(
        "hint: use --runs-root for fixture/demo data or create a real run first\n",
        style=style(TEXT_MUTED, bg=BG_WORKSPACE),
    )
    return workspace


def _append_stage_outputs(workspace: Text, outputs: tuple[str, ...]) -> None:
    if not outputs:
        workspace.append("○ not_checked\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        return
    for index, output in enumerate(outputs):
        marker_style = style(TEXT_SUCCESS if output.startswith("✓") else TEXT_MUTED, bg=BG_WORKSPACE)
        workspace.append(output, style=marker_style)
        workspace.append("\n" if index == len(outputs) - 1 else "  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))


def _status_grid(rows: list[tuple[str, str]]) -> Text:
    text = Text()
    left_width = 17
    value_width = 42
    for index in range(0, len(rows), 2):
        left_label, left_value = rows[index]
        right_label, right_value = rows[index + 1] if index + 1 < len(rows) else ("", "")
        text.append(f"{left_label.upper():<{left_width}} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
        text.append(_shorten(left_value, value_width).ljust(value_width), style=_value_style(left_value))
        if right_label:
            text.append("  ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
            text.append(f"{right_label.upper():<{left_width}} ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
            text.append(_shorten(right_value, value_width), style=_value_style(right_value))
        text.append("\n", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return text


def _value_style(value: str) -> str:
    if value.startswith("✓"):
        return style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    if value.startswith("?") or value.startswith("-") or value in {"unknown", "not_checked"}:
        return style(TEXT_MUTED, bg=BG_WORKSPACE)
    return style(TEXT_MUTED if value.startswith("○") else TEXT_MAIN, bg=BG_WORKSPACE)


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _section_box(title: str, lines: list[Text]) -> Text:
    box = Text()
    border_style = style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True)
    box.append("╭─ ", style=border_style)
    box.append(_shorten(title, BOX_WIDTH - 6), style=border_style)
    box.append(" " + "─" * max(0, BOX_WIDTH - len(title) - 5) + "╮\n", style=border_style)
    for line in lines:
        _append_box_line(box, line)
    box.append("╰" + "─" * BOX_WIDTH + "╯\n", style=border_style)
    return box


def _append_box_line(box: Text, line: Text) -> None:
    content = _truncate_text(line, BOX_WIDTH - 2)
    box.append("│ ", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))
    box.append_text(content)
    visible_len = len(content.plain)
    box.append(" " * max(0, BOX_WIDTH - visible_len - 1), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    box.append("│\n", style=style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True))


def _position_grid(fields: tuple[tuple[str, str], ...]) -> list[Text]:
    rows: list[Text] = []
    for start in range(0, len(fields), 4):
        group = fields[start : start + 4]
        rows.append(_cell_row(group, labels=True))
        rows.append(_cell_row(group, labels=False))
        if start + 4 < len(fields):
            rows.append(_blank_position_row())
    return rows


def _cell_row(fields: tuple[tuple[str, str], ...], *, labels: bool) -> Text:
    line = Text()
    cell_width = 29
    for index, (label, value) in enumerate(fields):
        if index:
            line.append(" │ ", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
        text = label.upper() if labels else value
        cell_style = style(TEXT_LABEL, bg=BG_WORKSPACE, bold=True) if labels else _value_style(value)
        line.append(f"{_shorten(text, cell_width):<{cell_width}}", style=cell_style)
    return line


def _blank_position_row() -> Text:
    line = Text()
    return line


def _status_for_job(state_label: str, output_status: str) -> str:
    lower = f"{state_label} {output_status}".lower()
    if "missing" in lower:
        return "missing"
    if "ready" in lower:
        return "ready"
    if "present" in lower or "passed" in lower or "finished" in lower:
        return "finished"
    if "waiting" in lower or "queued" in lower:
        return "queued"
    if "plan" in lower:
        return "read-only"
    return "unknown"


def _job_status_style(state_label: str, output_status: str) -> str:
    status = _status_for_job(state_label, output_status)
    if status in {"finished", "ready", "read-only"}:
        return style(TEXT_SUCCESS, bg=BG_WORKSPACE, bold=True)
    if status == "missing":
        return style(TEXT_MUTED, bg=BG_WORKSPACE)
    return style(TEXT_MUTED, bg=BG_WORKSPACE)


def _truncate_text(text: Text, limit: int) -> Text:
    if len(text.plain) <= limit:
        return text
    truncated = Text()
    remaining = max(0, limit - 1)
    for span_text, span_style in _text_spans(text):
        if remaining <= 0:
            break
        chunk = span_text[:remaining]
        truncated.append(chunk, style=span_style)
        remaining -= len(chunk)
    truncated.append("…", style=style(TEXT_MUTED, bg=BG_WORKSPACE))
    return truncated


def _pad_text(text: Text, width: int) -> Text:
    padded = Text()
    padded.append_text(text)
    padded.append(" " * max(0, width - len(text.plain)), style=style(TEXT_MAIN, bg=BG_WORKSPACE))
    return padded


def _text_spans(text: Text) -> list[tuple[str, str | None]]:
    spans: list[tuple[str, str | None]] = []
    plain = text.plain
    if not text.spans:
        return [(plain, None)]
    position = 0
    for span in sorted(text.spans, key=lambda item: item.start):
        if span.start > position:
            spans.append((plain[position : span.start], None))
        spans.append((plain[span.start : span.end], span.style))
        position = span.end
    if position < len(plain):
        spans.append((plain[position:], None))
    return spans


def _scene_value(scene: object, key: str, default: str) -> str:
    if isinstance(scene, dict):
        return str(scene.get(key) or default)
    return str(getattr(scene, key, default) or default)


def _scene_optional(scene: object, key: str) -> str | None:
    if isinstance(scene, dict):
        value = scene.get(key)
    else:
        value = getattr(scene, key, None)
    if value in (None, ""):
        return None
    return str(value)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else "missing"
