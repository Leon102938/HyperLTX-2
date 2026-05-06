# HANDOFF.md

## Stand 2026-05-05 Creative OS CLI Cockpit V1.6 Abschluss
- Read-only CLI Cockpit ist auf V1.6 finalisiert: `scripts/creative_os_status.py --style rich` zeigt ein Rich-Grid mit Header, linker Sidebar, Active Workspace, Scene Jobs und Bottom Panels.
- `--style plain` bleibt stabil; `--style rich` faellt bei fehlendem Rich sauber auf plain zurueck.
- Beispielrun: `creative-os-jungle-001`, Status `ready_for_ltx_i2v_takes`, Stage 01-08 passed, Stage 09 pending, Issues `none blocking`.
- Snapshots liegen unter `/workspace/cli_cockpit_snapshots/`: `overview_rich_cli.txt`, `all_rich_cli.txt`, `overview_plain_cli.txt`.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 12 Tests OK.
- Bewusst nicht gebaut: Stage 8, Render, LTX-Ausfuehrung, Video, Backend-Aufruf, n8n, API, neue Creative-OS-Stages, Textual.
- Naechster enger Schritt: Design visuell vom Operator pruefen.

## Stand 2026-05-05 Creative OS CLI Dashboard V1
- Read-only Dashboard ist gebaut: `scripts/creative_os_status.py`.
- Views: `overview`, `skills`, `stages`, `artifacts`, `issues`, `next`, `all`.
- Beleg: `python3 /workspace/scripts/creative_os_status.py --job-id creative-os-jungle-001 --view overview` zeigt `ready_for_stage_8`, Stage 01-08 passed, Stage 09 pending, none blocking.
- Das Tool liest nur Artefakte; keine Backend-/API-/Qwen-/Render-Aufrufe und keine Mutation der Run-Artefakte.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 5 Tests OK.
- Naechster enger Schritt bleibt: Stage 8 als kontrollierten LTX I2V Render-Plan/Executor-Gate entwerfen.

## Stand 2026-05-05 Creative OS Stage 7
- Stage 7 ist gebaut unter `agent_core/creative_os/`: LTX Motion Prompt Compiler und Stage-7-Runner.
- Entry: `python3 /workspace/scripts/creative_os_ltx_prompts.py --job-id creative-os-jungle-001`.
- Realer Lauf erzeugte `ltx_motion_prompts.json`, `ltx_prompt_audit.json` und `creative_os_stage7_report.md`.
- Audit-Stand: `scene_01=passed`, `scene_02=passed`, `scene_03=passed`, overall `passed`, `render_started=false`.
- Es wurde kein LTX-Render, kein Video, kein Take-Review, kein Assembly, kein n8n, keine API und kein Batch-System gebaut.
- Naechster enger Schritt: Stage 8 als kontrollierten LTX I2V Render-Plan/Executor-Gate entwerfen.

## Stand 2026-05-05 Creative OS Stage 6
- Stage 6 ist gebaut unter `agent_core/creative_os/`: Keyframe-Generator, heuristic Keyframe-QA und Stage-6-Runner.
- Entry: `python3 /workspace/scripts/creative_os_keyframes.py --job-id creative-os-jungle-001 --review-provider heuristic`.
- Realer Lauf erzeugte 3 echte PNGs unter `/workspace/agent_runs/creative-os-jungle-001/creative_os/keyframes/`.
- Artefakte: `keyframe_manifest.json`, `keyframe_review.json`, `keyframe_generation_log.json`, `creative_os_stage6_report.md`.
- QA-Stand: `scene_01=passed`, `scene_02=passed` nach Stage 6.1 manual-structured Review, `scene_03=passed`.
- Es wurde kein LTX Motion Prompt Compiler, kein LTX Render, kein Video, kein n8n, keine API und kein Batch-System gebaut.

## Stand 2026-05-05 Creative OS V1 Dry-Run
- Andockbare Creative OS V1 Dry-Run-Schicht ist isoliert unter `agent_core/creative_os/` gebaut; Entry ist `scripts/creative_os_dry_run.py`.
- Der Pfad stoppt bei `zimage_prompts.json`: kein Bildrender, kein LTX-Render, kein Qwen-VL, kein Batch, kein n8n, keine Runtime-Aenderung.
- Beleglauf: `/workspace/agent_runs/creative-os-jungle-001/creative_os/` mit allen Pflichtartefakten und 3 Z-Image-Prompt-Objekten.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_skill_loader.py /workspace/tests/test_creative_os_runner.py /workspace/tests/test_creative_os_prompts.py -v` -> 4 Tests OK.
- Naechster enger Schritt: Creative-OS-Artefakte manuell auditieren und erst danach entscheiden, ob der bestehende Stop-after-`model_prompts`-Pfad diese Schicht aufrufen soll.

## Stand 2026-05-01 Tagesabschluss
- Phase F2 Grundlage ist umgesetzt: `agent_core/creative_system/` enthaelt Hook Patterns, Shot Recipes und Anti-Patterns; Morning Reset nutzt feste Shot Recipes und Hook Functions.
- Backend Prompt Policy ist aktiv: Z-Image bekommt fuer Morning Reset positive-only Prompts; LTX bekommt positive Prosa plus kurze Avoid-Liste.
- Pro geplantem Run gibt es jetzt `prompt_audit.json` und `model_prompts.json`. Letzteres zeigt `zimage_prompt_sent`, `ltx_prompt_sent`, Prompt-Quellen und Leak Checks.
- CLI Live Dashboard ist verbessert: TTY-Redraw mit `--live`/`--no-live`, Current Work, Prompt Preview, Pipeline, Szenen und Artefaktstatus; Non-TTY bleibt Append-Ausgabe.
- Aktueller Dry-Run: `/workspace/agent_runs/phase-f2-creative-os-dry-run` mit gruenem Prompt Audit und Model Prompt Trace. Kein echter Render wurde gestartet.
- Tagesabschluss-Tests: 70 Unit Tests OK.

## Stand 2026-04-30
- `init.sh` ist klein und stabilisiert: normaler HF-Downloader als Default, Xet aus, minimaler Init-Lock, Qwen3-VL optional.
- LTX/Gemma ist wieder lauffaehig: globale Main-Runtime nutzt `transformers 4.52.4`.
- Qwen3-VL Review ist isoliert: `/workspace/venvs/qwen3-vl-review` mit `transformers 5.7.0` und `kernels 0.13.0`, aufgerufen ueber `/workspace/scripts/qwen3_vl_review_subprocess.py`.
- Die Qwen3-VL-Venv wird nicht archiviert. Sie wird nach Restore mit `/workspace/scripts/ensure_qwen3_vl_review_runtime.sh` neu erstellt.
- Phase E2/E2.1/E2.2 CLI Produktions-Cockpit ist umgesetzt; `scripts/agent_core_cli.py --inspect-run <job_id>` ist der schnelle Diagnosepfad mit Pipeline Labels, Vision-Status, gruppierten Issues und Next Actions.
- Erster Morning-Reset-Quality-Fix ist umgesetzt: Visual Prompt Sanitizer, Safe Morning Reset Motifs, allowed_props Cleanup, Storyboard Prompt Schutz und strengere Device-/UI-Risiken.
- Aktueller echter Kontrollrun: `quality-morning-reset-006`, technisch `success=True`, `final_phase=assembled`, `final.mp4` vorhanden, aber Final Quality `failed`.
- Diagnose `quality-morning-reset-006`: Scene 1 Fake-Text, Scene 2 Smartphone/Phone neben Glas in einem Take-Kontext, Scene 3 Split-Screen/Collage/Text/UI-Drift, Qwen3-VL non-json/parser warning.
- Offener Bug fuer morgen: rejected Take darf nicht selected werden, wenn passed/needs_review existiert; zusaetzlich hartes Keyframe Gate gegen Fake-Text/Phone/Split-Screen und robustere Qwen3-VL JSON-Auswertung.

## Restore Nach Frischem Pod
1. Archiv nach `/workspace` entpacken.
2. `bash /workspace/init.sh`
3. `bash /workspace/scripts/ensure_qwen3_vl_review_runtime.sh`
4. FastAPI/Director pruefen:
   - `curl -sS http://127.0.0.1:8000/health`
   - `curl -sS http://127.0.0.1:8011/v1/models`
5. Schneller Run-Check:
   - `python3 /workspace/scripts/agent_core_cli.py --inspect-run quality-morning-reset-006`
   - `python3 /workspace/scripts/agent_core_cli.py --inspect-run quality-morning-reset-005`

## Naechster Schritt
Zuerst `/workspace/agent_runs/phase-f2-creative-os-dry-run/prompt_audit.json` und `model_prompts.json` pruefen. Nur wenn die Backend-Prompts sauber sind, `quality-morning-reset-009` manuell starten.
