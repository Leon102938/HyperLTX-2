from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from rich.text import Text
from textual.widgets import Static

Region = Literal[
    "header",
    "sidebar_top",
    "sidebar_bottom",
    "main",
    "bottom_left",
    "bottom_mid",
    "bottom_right",
    "bottom_far_right",
]


class CockpitPanel(Static):
    def __init__(self, title: str, body: Text, *, panel_id: str, classes: str | None = None) -> None:
        super().__init__(body, id=panel_id, classes=classes)
        self.border_title = title


@dataclass(frozen=True)
class PanelDefinition:
    panel_id: str
    title: str
    purpose: str
    required_data: tuple[str, ...]
    render: Callable[[object], Text]
    optional: bool
    default_region: Region
    widget_id: str
    classes: str | None = None


@dataclass(frozen=True)
class PanelConfig:
    enabled: bool
    region: Region
