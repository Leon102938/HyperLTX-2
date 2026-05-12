---
name: textual-cockpit-ui
description: Use this skill whenever working on the Creative OS Textual cockpit, active workspace panels, stage panels, pipeline map, panel visual fidelity, Textual TUI layout, borders, cards, colors, and visual reference matching.
---

# Textual Cockpit UI Skill

## Goal
Build and polish the Creative OS Textual cockpit panels so they match the visual references under:
/workspace/codex/panel_build/01_VISUAL_REFERENCES/pipeline_panels/

## Hard Rules
- Do not redesign the whole cockpit.
- Do not change Header, System Status, Bottom Panels, or Pipeline Map unless explicitly requested.
- Preserve dark blue/black cyber look.
- Preserve cyan borders.
- Use existing box/border helpers.
- Do not introduce broken borders.
- Avoid long text overflow.
- Prefer compact labels and truncated previews.
- Every panel must be stable in terminal width.
- No fake live values as real values.
- Fixture/demo values must be marked as fixture/demo.

## Required Workflow
1. Read the target panel visual_analysis.md.
2. Inspect the target reference.png.
3. Compare current panel against reference.
4. Make only the minimal UI change.
5. Run cockpit tests.
6. Update panel_build progress docs.
7. Report what was changed and what was not.

## Visual Matching Rules
For every target panel:
- match block order
- match inner card structure
- match status logic
- match color meaning
- match density
- keep text short
- no huge prompt walls
- no unrelated panels changed

## Test Commands
Run:
python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v

If quick:
python3 -m unittest /workspace/tests/test_creative_os_status.py -v