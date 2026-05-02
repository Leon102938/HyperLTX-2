# SESSION_SUMMARY.md

## Datum UTC
- Erstprecheck: 2026-05-01T10:41:10Z
- Archivzeit: 2026-05-01T10:58:47Z

## Finaler Projektstand
- Phase F1.1 Prompt Compiler Cleanup war bereits abgeschlossen: positive, negative und kombinierte Model-Prompts sind getrennt.
- Heute abgeschlossen: F2 Creative Operating System Grundlage, Backend Prompt Trace, Backend Prompt Policy und produktiveres CLI Live Dashboard.
- Kein echter Render, kein GPU-Render, kein Model-Download, kein Runtime-/Dependency-Upgrade und kein `init.sh`-Umbau wurden gestartet.

## Heutige Phasen
- F1.1 Prompt Compiler Cleanup: bestehender sauberer Audit-Stand beibehalten.
- F2 Creative OS Grundlage: Hook Patterns, Shot Recipes und Anti-Patterns ergänzt.
- CLI Live Dashboard Verbesserung: TTY-Redraw mit `--live`/`--no-live`, Current Work, Prompt Preview, Pipeline, Szenen und Artefakte.
- Prompt Trace/model_prompts: `model_prompts.json` pro geplantem Run mit Z-Image-/LTX-Prompts und Leak Checks.
- Backend Prompt Policy: Z-Image positive-only, LTX positive-plus-short-avoid.

## Precheck
- `python3 --version`: Python 3.12.13.
- Global Transformers: 4.52.4 unter `/usr/local/lib/python3.12/dist-packages/transformers/__init__.py`.
- Qwen-Venv Transformers: 5.7.0 unter `/workspace/venvs/qwen3-vl-review/lib/python3.12/site-packages/transformers/__init__.py`.
- Qwen3-VL Import: `qwen3vl import ok`.
- FastAPI `/health`: `{"status":"ok","init_ready":true,"ltx_backend":"ltx-2.3"}`.
- Director `/v1/models`: Qwen GGUF Director Modell wurde gelistet.
- Git-Status war vor Beginn bereits dirty; bestehende fremde/alte Änderungen wurden nicht revertet.

## Dry-Runs
- Neuer Dry-Run: `/workspace/agent_runs/phase-f2-creative-os-dry-run`.
- Enthalten: `plan.json`, `scene_plan.json`, `storyboard_plan.json`, `prompt_audit.json`, `model_prompts.json`, `state.json`, `director_output.json`, `logs/agent.log`.
- `prompt_audit.json`: alle F2-/F1.1-Checks true.
- `model_prompts.json`: `zimage_positive_only_applied=true`, `ltx_short_avoid_applied=true`, keine Debuglabels oder Script-Snippets in Backend-Prompts.
- Z-Image Prompt Word Counts: Scene 1 61, Scene 2 29, Scene 3 57.
- LTX Prompt Word Counts: Scene 1 77, Scene 2 45, Scene 3 73.

## Tests
- Ausgeführt: `python3 -m unittest tests/test_creative_system.py tests/test_cli_live_dashboard.py tests/test_planner_rules.py tests/test_scene_planner.py tests/test_storyboard_pipeline.py tests/test_take_visual_review.py tests/test_output_quality_utils.py tests/test_final_quality_verdict.py`.
- Ergebnis: 70 Tests OK.
- Zusätzlich: `python3 -m py_compile` für geänderte Kernmodule und CLI OK.

## Wichtigste Offene Probleme
- Der nächste echte visuelle Beleg fehlt absichtlich noch, weil heute kein Render gestartet wurde.
- Morgen muss zuerst der F2-Dry-Run-Audit manuell gelesen werden, damit kein Prompt-Trace-Fehler in einen echten Render geht.
- Falls ein echter `quality-morning-reset-009` erneut fehlschlägt, ist zuerst `model_prompts.json` gegen tatsächliche Backend-Prompts und danach das sichtbare Video zu prüfen.

## Morgen Als Nächstes
1. `/workspace/agent_runs/phase-f2-creative-os-dry-run/prompt_audit.json` und `model_prompts.json` prüfen.
2. Nur wenn Backend-Prompts sauber sind, `quality-morning-reset-009` manuell starten.

## Restore-Anleitung
1. Archiv nach `/workspace` entpacken.
2. `bash /workspace/init.sh`
3. `bash /workspace/scripts/ensure_qwen3_vl_review_runtime.sh`
4. Services prüfen:
   - `curl -sS http://127.0.0.1:8000/health`
   - `curl -sS http://127.0.0.1:8011/v1/models`
5. Vor Render: `python3 /workspace/scripts/agent_core_cli.py --inspect-run phase-f2-creative-os-dry-run`

## Enthaltene Dateien
- Root/config: `init.sh` und vorhandene kleine Root-Konfigurationsdateien.
- `agent_core/` inklusive Adapter und `creative_system/`.
- Relevante `scripts/`.
- `tests/`.
- `codex/*.md`.
- Kleine Run-Artefakte aus F1/F1.1/F2 und optional kleine JSON-/Log-Artefakte aus Quality-Runs, ohne große Medienordner.

## Bewusst Nicht Enthalten
- `/workspace/models`
- `/workspace/venvs`
- `/workspace/LTX-2/checkpoints`
- Caches und HF cache
- `node_modules`
- `*.safetensors`, `*.gguf`, `*.incomplete`
- komplette `/workspace/jobs` Output-Ordner
- große Videos und Backend-Checkpoint-Ordner
