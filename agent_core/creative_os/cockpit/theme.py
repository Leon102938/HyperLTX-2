from __future__ import annotations

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
SCENE_CARD_INNER_WIDTH = 66
SCENE_CARD_LABEL_WIDTH = 9

COCKPIT_CSS = """
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
    width: 30%;
    min-width: 30;
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
    overflow-y: auto;
    overflow-x: hidden;
}

#workspace-content {
    width: 100%;
    height: auto;
    background: #0B1628;
    color: #E5E7EB;
}

#system-status {
    height: 6;
    border: round #38BDF8;
    padding: 0 1;
}

#pipeline-map {
    border: round #1E3A5F;
    height: 1fr;
    padding: 0 1;
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

#issues-tile.issues-none {
    border: round #38BDF8;
}

#issues-tile.issues-warning {
    border: round #FBBF24;
}

#issues-tile.issues-error {
    border: round #EF4444;
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


def style(color: str, *, bg: str | None = None, bold: bool = False) -> str:
    parts = []
    if bold:
        parts.append("bold")
    parts.append(color)
    if bg:
        parts.append(f"on {bg}")
    return " ".join(parts)
