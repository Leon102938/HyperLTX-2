# HyperLTX Project Session Summary

Datum UTC: 2026-04-28

## Letzter sicherer Projektstand
- HyperLTX-/Content-Maschine-Core bleibt funktional auf dem bestehenden FastAPI/agent_core-Pfad.
- Diese Session hat Output-Quality-Planungslogik erweitert: Phase A, Phase B1 und Phase B2.
- Keine Runtime-, Backend-, API-, llama.cpp-, init/start- oder GUI-Aenderung wurde fuer Phase B2 vorgenommen.

## Phase A Ergebnis
- Scene World Contract + PromptBuilder v2 umgesetzt.
- `scene_world_contract` wird in Plan-/Scene-/Take-Metadaten sichtbar.
- Social-Tip-Prompts enthalten harte Verbote gegen readable text, handwriting, paper/notebook/document pages, screens/UI, labels/logos/posters/signs, typography/glyphs/letters/numbers.

## Phase B1 Ergebnis
- Storyboard-/Keyframe-Prompts sind scene-specific und contract-aware.
- `build_storyboard_render_plan()` speichert pro Kandidat `effective_prompt`, `prompt_source`, `candidate_prompt_text`, `scene_prompt_text`, `scene_world_contract` und `storyboard_prompt_metadata`.
- `ZImageStoryboardAdapter` bevorzugt `storyboard_step.params["effective_prompt"]` vor Candidate-/Global-Fallbacks.

## Phase B2 Ergebnis
- Lightweight Keyframe Visual Risk Review eingefuehrt.
- `evaluate_keyframe_visual_risk()` klassifiziert Kandidaten als `passed`, `needs_review` oder `rejected`.
- Review-Metadaten: `risk_score`, `issues`, `warnings`, `policy_version`, `source`, `checked_contract_fields`, `checked_prompt_fields`.
- Storyboard-Auswahl bevorzugt technisch valide Kandidaten in der Reihenfolge `passed` vor `needs_review` vor `rejected`.
- Kein OCR, keine Vision-LLM-Analyse und keine finale Bildqualitaetsgarantie.

## Tests und Ergebnis
- `python3 -m unittest tests/test_planner_rules.py` OK
- `python3 -m unittest tests/test_output_quality_utils.py` OK
- `python3 -m unittest tests/test_storyboard_pipeline.py` OK
- `python3 -m unittest tests/test_scene_planner.py` OK
- `python3 -m unittest tests/test_assembler_mux.py` OK

## Real geaenderte Projektdateien
- `agent_core/adapters/zimage_storyboard_adapter.py`
- `agent_core/agent.py`
- `agent_core/director.py`
- `agent_core/planner.py`
- `agent_core/prompt_builder.py`
- `agent_core/schemas.py`
- `agent_core/utils.py`
- `tests/test_output_quality_utils.py`
- `tests/test_planner_rules.py`
- `tests/test_scene_planner.py`
- `tests/test_storyboard_pipeline.py`
- `codex/ACTIVE_PLAN.md`
- `codex/CHANGELOG.md`
- `codex/HANDOFF.md`
- `codex/MEMORY.md`
- `codex/PROJECT_STATE.md`
- Vorhandene projektbezogene Startup-/Director-Dateien aus frueherem Fortschritt wurden ins Archiv aufgenommen, aber Phase B2 hat sie nicht geaendert: `init.sh`, `scripts/agent_core_cli.py`, `scripts/check_director_llm.py`, `scripts/download_director_model.py`, `scripts/ensure_llama_cpp.sh`, `scripts/serve_director_llm.sh`.

## Erzeugte Dry-Runs
- `/workspace/agent_runs/phaseA-verify-morning-reset` falls vorhanden
- `/workspace/agent_runs/phaseA-verify-focus-break` falls vorhanden
- `/workspace/agent_runs/phase-b1-dry-morning-reset`
- `/workspace/agent_runs/phase-b1-dry-focus-break`
- `/workspace/agent_runs/phase-b2-dry-morning-reset`
- `/workspace/agent_runs/phase-b2-dry-focus-break`

## Offene naechste Phasen
- Phase C: Take Visual Review / Postability Score
- Phase D: Final Quality Verdict
- Phase E: CLI Produktions-Cockpit

## Explizite Nicht-Ziele / Nicht enthalten
- Keine Runtime-/Backend-/llama.cpp-Aenderung in Phase B2.
- Codex-CLI-Fix-Dateien sind nicht Teil des HyperLTX-Projektarchivs.
- Nicht archiviert: `/workspace/codex_cli_fix`, `codex/CODEX_TERMINAL_DIAGNOSIS.md`, `scripts/start_codex_stable.sh`, Modelle, Checkpoints, node_modules, npm cache, komplette llama.cpp Runtime.
