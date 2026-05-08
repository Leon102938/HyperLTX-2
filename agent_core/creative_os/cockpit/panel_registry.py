from __future__ import annotations

from agent_core.creative_os.cockpit.panel_types import PanelConfig, PanelDefinition
from agent_core.creative_os.cockpit.panels import (
    active_workspace_panel,
    artifacts_panel,
    issues_panel,
    next_panel,
    pipeline_map_panel,
    skill_health_panel,
    system_status_panel,
)

PANEL_CONFIG: dict[str, PanelConfig] = {
    "header": PanelConfig(enabled=True, region="header"),
    "system_status": PanelConfig(enabled=True, region="sidebar_top"),
    "pipeline_map": PanelConfig(enabled=True, region="sidebar_bottom"),
    "active_workspace": PanelConfig(enabled=True, region="main"),
    "skill_health": PanelConfig(enabled=True, region="bottom_left"),
    "artifacts": PanelConfig(enabled=True, region="bottom_mid"),
    "issues": PanelConfig(enabled=True, region="bottom_right"),
    "next": PanelConfig(enabled=True, region="bottom_far_right"),
}

PANEL_REGISTRY: dict[str, PanelDefinition] = {
    "system_status": PanelDefinition(
        panel_id="system_status",
        title="SYSTEM STATUS",
        purpose="Show local read-only backend and capability status for the fixture/session.",
        required_data=("system_status",),
        render=system_status_panel.render,
        optional=False,
        default_region="sidebar_top",
        widget_id="system-status",
        classes="panel",
    ),
    "pipeline_map": PanelDefinition(
        panel_id="pipeline_map",
        title="PIPELINE MAP",
        purpose="Show the current Creative OS stage map without starting pipeline work.",
        required_data=("pipeline_map",),
        render=pipeline_map_panel.render,
        optional=False,
        default_region="sidebar_bottom",
        widget_id="pipeline-map",
        classes="panel",
    ),
    "active_workspace": PanelDefinition(
        panel_id="active_workspace",
        title="ACTIVE WORKSPACE",
        purpose="Show the current operator focus, stage output and scene jobs.",
        required_data=("workspace",),
        render=active_workspace_panel.render,
        optional=False,
        default_region="main",
        widget_id="workspace",
        classes=None,
    ),
    "skill_health": PanelDefinition(
        panel_id="skill_health",
        title="SKILL HEALTH",
        purpose="Summarize loaded, fallback and missing skills.",
        required_data=("skill_health",),
        render=skill_health_panel.render,
        optional=False,
        default_region="bottom_left",
        widget_id="skill-tile",
        classes="tile",
    ),
    "artifacts": PanelDefinition(
        panel_id="artifacts",
        title="ARTIFACTS",
        purpose="Show expected Creative OS artifacts from the read-only inspection.",
        required_data=("artifacts",),
        render=artifacts_panel.render,
        optional=False,
        default_region="bottom_mid",
        widget_id="artifacts-tile",
        classes="tile",
    ),
    "issues": PanelDefinition(
        panel_id="issues",
        title="ISSUES",
        purpose="Surface blocking issues if present.",
        required_data=("issues",),
        render=issues_panel.render,
        optional=True,
        default_region="bottom_right",
        widget_id="issues-tile",
        classes="tile issues-none",
    ),
    "next": PanelDefinition(
        panel_id="next",
        title="NEXT",
        purpose="Show the next technical/operator step without executing it.",
        required_data=("next_panel",),
        render=next_panel.render,
        optional=False,
        default_region="bottom_far_right",
        widget_id="next-tile",
        classes="tile",
    ),
}


def enabled_panels() -> dict[str, PanelDefinition]:
    return {panel_id: panel for panel_id, panel in PANEL_REGISTRY.items() if PANEL_CONFIG[panel_id].enabled}
