from __future__ import annotations

import textwrap

from rich.text import Text

from agent_core.creative_os.cockpit.theme import (
    BG_SCENE_CARD,
    BORDER_PRIMARY,
    SCENE_CARD_INNER_WIDTH,
    SCENE_CARD_LABEL_WIDTH,
    TEXT_ACTIVE,
    TEXT_LABEL,
    TEXT_MAIN,
    TEXT_MUTED,
    TEXT_SUCCESS,
    style,
)


def rows_text(rows: tuple[tuple[str, str], ...] | list[tuple[str, str]], *, label_width: int = 16, bg: str | None = None) -> Text:
    text = Text()
    for label, value in rows:
        text.append(f"{label.upper():<{label_width}} ", style=style(TEXT_LABEL, bg=bg, bold=True))
        text.append(f"{value}\n", style=value_style(value, bg=bg))
    return text


def value_style(value: str, *, bg: str | None = None) -> str:
    if value.startswith("✓"):
        return style(TEXT_SUCCESS, bg=bg, bold=True)
    if value.startswith("?") or value.startswith("-"):
        return style(TEXT_MUTED, bg=bg)
    if "Stage 09" in value or value == "paused":
        return style(TEXT_ACTIVE, bg=bg, bold=True)
    return style(TEXT_MAIN, bg=bg)


def status_text_style(status: str) -> str:
    if status in {"passed", "done"}:
        return TEXT_SUCCESS
    if status == "pending":
        return "#94A3B8"
    if status == "running":
        return "#FBBF24"
    if status == "needs_review":
        return TEXT_ACTIVE
    if status == "unknown":
        return TEXT_MUTED
    if status in {"rejected", "error", "failed", "missing"}:
        return "#EF4444"
    if status == "paused":
        return TEXT_ACTIVE
    return TEXT_MAIN


def scene_card_text(scene: object) -> Text:
    scene_id = _scene_value(scene, "scene_id", "unknown")
    keyframe = _scene_value(scene, "keyframe", "")
    summary = _scene_value(scene, "summary", "motion prompt present")
    state_label = _scene_value(scene, "state_label", "READY FOR LTX")
    status = _scene_value(scene, "status", "waiting for Stage 09 render gate")
    card = Text()
    border_style = style(BORDER_PRIMARY, bg=BG_SCENE_CARD)
    card.append("╭" + "─" * SCENE_CARD_INNER_WIDTH + "╮\n", style=border_style)
    _append_card_line(
        card,
        f"[✓] {_shorten_for_card(scene_id, 16)}  {_shorten_for_card(state_label, 18)}",
        style=style(TEXT_SUCCESS, bg=BG_SCENE_CARD, bold=True),
    )
    _append_card_field(card, "keyframe", _basename(keyframe), style=style(TEXT_MAIN, bg=BG_SCENE_CARD), max_lines=1)
    _append_card_field(card, "motion", summary, style=style(TEXT_MUTED, bg=BG_SCENE_CARD), max_lines=1)
    _append_card_field(card, "status", status, style=style(TEXT_ACTIVE, bg=BG_SCENE_CARD), max_lines=1)
    card.append("╰" + "─" * SCENE_CARD_INNER_WIDTH + "╯\n", style=border_style)
    return card


def _append_card_field(card: Text, label: str, value: str, *, style: str, max_lines: int) -> None:
    value_width = SCENE_CARD_INNER_WIDTH - SCENE_CARD_LABEL_WIDTH - 3
    lines = _wrap_for_card(value, value_width, max_lines=max_lines)
    label_text = f"{label}:"
    for index, line in enumerate(lines):
        prefix = f" {label_text:<{SCENE_CARD_LABEL_WIDTH}} " if index == 0 else f" {'':<{SCENE_CARD_LABEL_WIDTH}} "
        _append_card_segments(card, [(prefix, style_label(index)), (line, style)])


def style_label(index: int) -> str:
    return style(TEXT_LABEL if index == 0 else TEXT_MAIN, bg=BG_SCENE_CARD)


def _append_card_line(card: Text, value: str, *, style: str) -> None:
    _append_card_segments(card, [(f" {_shorten_for_card(value, SCENE_CARD_INNER_WIDTH - 1)}", style)])


def _append_card_segments(card: Text, segments: list[tuple[str, str]]) -> None:
    border_style = style(BORDER_PRIMARY, bg=BG_SCENE_CARD)
    content_len = sum(len(text) for text, _style_value in segments)
    card.append("│", style=border_style)
    for text, segment_style in segments:
        card.append(text, style=segment_style)
    card.append(" " * max(0, SCENE_CARD_INNER_WIDTH - content_len), style=style(TEXT_MAIN, bg=BG_SCENE_CARD))
    card.append("│\n", style=border_style)


def _wrap_for_card(value: str, width: int, *, max_lines: int) -> list[str]:
    compact = " ".join(str(value).split())
    if not compact:
        return [""]
    lines = textwrap.wrap(compact, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return [""]
    if len(lines) <= max_lines:
        return [line.ljust(width) for line in lines]
    kept = lines[:max_lines]
    kept[-1] = _shorten_for_card(" ".join([kept[-1], *lines[max_lines:]]), width)
    return [line.ljust(width) for line in kept]


def _shorten_for_card(value: str, max_chars: int) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 1)].rstrip() + "…"


def _shorten(text: object, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else "missing"


def _scene_value(scene: object, key: str, default: str) -> str:
    if isinstance(scene, dict):
        return str(scene.get(key) or default)
    return str(getattr(scene, key, default) or default)
