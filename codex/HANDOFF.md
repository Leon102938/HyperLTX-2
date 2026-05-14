# HANDOFF.md

## 2026-05-14 Phase 1 Live Cockpit Orchestration
- V3-Fix: Phase-1-Runs sperren Stage `10` bis `15` in der Pipeline Map als out-of-scope/pending; vorhandene Alias-Artefakte wie `stage6_review_decision.json` koennen Stage `10` nicht mehr gruen faerben.
- Stage `09` Active Workspace zeigt jetzt echte Manifest-/Live-Daten sichtbar: Backend Status/Reason, Overall Status, Live Stage 09, Finished Files und Gallery.
- Watch-Refresh rendert nicht mehr bei reiner `last_refresh_time`-Aenderung; manuelle Stage-Auswahl bleibt bei Watch erhalten.
- Stage `09` schreibt `keyframe_manifest.json` waehrend der echten Image-Job-Schleife fort, damit Watch running/progress/finished/error aus realem Backend-State lesen kann.
- V3-Smokes: `/workspace/agent_runs/live-v3-smoke-noimages-20260514/creative_os` und `/workspace/agent_runs/live-v3-smoke-images-20260514/creative_os`.
- V2-Fix: Missing/disabled Backend markiert Stage `09` im `live_status.json` nicht mehr als `done`, sondern als `error`; `completed_stages` bleibt dann `00` bis `08`, passend zu `phase1_status.json`.
- `stage_events.jsonl` schreibt bei disabled Backend `09 running -> 09 error`, nicht mehr `09 done`.
- Pipeline Map nutzt Live-Stage-Status vor Artifact-Anwesenheit: Stage `09` wird bei `paused_missing_backend` rot/warnend statt gruen, auch wenn `keyframe_manifest.json` existiert.
- `--open-cockpit` startet Textual nicht mehr als Background-Prozess im selben TTY. Der Flag gibt zwei sichere Terminal-Befehle aus und beendet sauber; ohne TTY kein OSError.
- Neuer Debug-Parameter: `--stage-delay-seconds 0.5` fuer sichtbare Live-Schritte; Default `0`.
- Image-Flags: `--generate-images` fuer echte Z-Image-Jobs, `--no-generate-images` fuer disabled/no-image Live-Runs; `--no-images` bleibt kompatibler Alias.
- V2-Smokes: disabled `/workspace/agent_runs/live-v2-smoke-20260514/creative_os` mit Stage `09=error`; image `/workspace/agent_runs/live-v2-smoke-images-20260514/creative_os` mit 3 PNGs und Gallery.
- Neuer Live-CLI-Pfad: `python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1-live --job-id <id> --topic "..." --pipeline shortform_storyboard_v1 --mode visual_adventure --style cinematic_nature --format portrait --duration 9s --scenes 3 [--open-cockpit]`.
- Live-Runner schreibt waehrend des Runs `live_status.json` und `stage_events.jsonl` unter `/workspace/agent_runs/<job-id>/creative_os/`; Batch `creative-os run-phase1` bleibt kompatibel.
- Live-State trennt `viewed_stage` von `real_run_stage`/`current_running_stage`; Cockpit startet fuer Live-Runs bei Stage `00`, waehrend der echte Runner bis Stage `09` fortschreibt.
- Cockpit-Watch-Pfad: `python3 /workspace/scripts/creative_os_cockpit.py --job-id <id> --runs-root /workspace/agent_runs --watch --refresh-sec 1`.
- `--open-cockpit` startet nur stabil als separater Textual-Prozess, wenn eine TTY vorhanden ist; ohne TTY gibt die CLI exakt den Watch-Befehl aus.
- Real-Data-Regel bleibt eng: fehlende Artefakte werden `missing`, fehlendes Stage-09-Manifest zeigt keine Demo-Cards, und Real-Runs zeigen Progress nur aus echten Manifestwerten.
- Stage `09` liest weiter nur `keyframe_manifest.json`: queued/running/finished/error, elapsed, output_path, file_exists/size/mtime und gallery_path; ETA bleibt `unavailable`/nicht angezeigt, wenn nicht ableitbar.
- Smoke-Run: `/workspace/agent_runs/live-smoke-20260514/creative_os`.
- Tests gruen: `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v` und `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`.
- Nicht gebaut: Stage `10-15` Runtime, LTX, n8n/API, Redesign, neue Textual-Version.

## 2026-05-13 Phase 1 Hardening / Stage 09 Retry
- Frischer E2E-Run: `/workspace/agent_runs/phase1-hardening-smoke-20260513/creative_os`.
- Run-CLI:
  `python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1 --job-id phase1-hardening-smoke-20260513 --topic "jungle safari at sunrise" --pipeline shortform_storyboard_v1 --mode visual_adventure --style cinematic_nature --format portrait --duration 9s --scenes 3`
- Retry-CLI:
  `python3 /workspace/scripts/agent_core_cli.py creative-os retry-keyframes --job-id <job-id> --runs-root /workspace/agent_runs [--scene scene_02] [--dry-run] [--force]`
- Retry liest nur `keyframe_manifest.json`. Kandidaten sind Jobs mit `failed/error/queued/running`, fehlender `output_path` oder `file_exists=false`.
- Retry schreibt nur `keyframe_manifest.json`, `phase1_status.json` und ggf. `keyframe_gallery.html`; Stage `00` bis `08` bleiben unveraendert.
- Ohne `--force` werden fertige vorhandene PNGs nicht neu erzeugt. `--force --scene scene_02 --dry-run` plant genau diese Szene.
- Smoke: `phase1-hardening-retry-sim-20260513` hatte fehlende `scene_02.png`; `retry-keyframes --scene scene_02` hat sie neu erzeugt und Status wieder auf `finished` gesetzt.
- Cockpit Stage `09` zeigt echte Preview-Pfade, Datei-Status, Gallery und bei fehlendem Manifest `missing manifest` statt Fake-Cards.
- Tests gruen: Cockpit 16, Status 21. Textual bleibt `0.89.1`.
- Nicht gebaut: Stage `10-15`, LTX Video, Assembly, Final Output, n8n/API, Redesign, neue Dependencies.

## 2026-05-13 Phase 1 Runtime bis Stage 09
- Neuer CLI-Pfad: `python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1 --job-id <id> --topic "..." --pipeline shortform_storyboard_v1 --mode visual_adventure --style cinematic_nature --format portrait --duration 9s --scenes 3`.
- Der Runner schreibt lokale Creative-OS-Artefakte nach `/workspace/agent_runs/<job-id>/creative_os/` fuer Stage `00` bis `09`.
- Stage-09-Artefakt ist `keyframe_manifest.json` mit pro Szene `scene_id`, `prompt`, `backend`, `status`, `output_path`, `progress_percent`, `elapsed` und `error`.
- Wenn Z-Image ueber den bestehenden HTTP-Backendpfad nicht erreichbar ist, meldet der Run `phase1_paused_missing_image_backend`; keine Fake-Bilder werden erzeugt oder als Erfolg markiert.
- Cockpit/Inspector lesen `keyframe_manifest.json` jetzt als echte Stage-09-Jobquelle; Fixture/Demo bleibt kompatibel.
- Nicht gebaut: Stage `10` bis `15` Runtime, LTX Video, Assembly, Final Output, n8n/API.
- Tests: `test_creative_os_cockpit.py` gruen, `test_creative_os_status.py` gruen inklusive Phase-1-CLI-/Missing-Backend-Test und Textual-0.89.x-Pin.

## 2026-05-09 Cockpit Panel Completion V0.2
- Active Workspace Stage-Panels 00-15 sind als read-only Operator-Oberflaechen vorhanden.
- Stage 00 enthaelt den read-only Command Composer mit sichtbaren Eingabe-/Preview-Feldern und dem deaktivierten `Run planned / disabled in V0.2` Aktionshinweis; er startet keine Commands.
- 04-08 decken Strategy, Beat/Hook, Judge, Scene Contracts und Image Prompt Compiler mit stage-spezifischen Feldern ab; 10-15 decken Keyframe Review, LTX Motion Prompts, Video Generation, Video Review, Assembly und Final Output ab.
- Stage 09 wurde nicht neu redesignt und bleibt das Image-Job-Card-Panel mit Preview-Slots, Expanded Image 2 und Unicode-Progressbar.
- Snapshots: `/workspace/cockpit_snapshots_2026-05-09_v02/`; Archive: `/workspace/session_archive_2026-05-09_cockpit_presprint/` und `/workspace/session_archive_2026-05-09_cockpit_final/`.
- Verifikation: Cockpit-Tests und Status-Tests gruen; Fixture- und Missing-Run-Startchecks ohne Traceback.
- Nicht gebaut: Render, API/n8n, neue Runs, Pipeline-Integration oder Command Execution.

## 2026-05-09 Textual Cockpit Stage 09 Image Jobs Pass
- Nur Stage 09 im Active Workspace weitergebaut: `PROMPTS / IMAGE JOBS` besteht jetzt aus einzelnen Image-Job-Bloecken mit Preview-Slot, Job/Prompt, Status und Expand-Pfeil.
- In Stage 09 waehlt j/k den Image Job, Enter/Space toggelt Expand/Collapse. Pfeil hoch/runter bleibt fuer Stage-Auswahl.
- Fixture-Demo: scene_01 ready mit Preview, scene_02 generating mit `demo progress` 62%, elapsed/backend, scene_03 in queue.
- Verifikation: Cockpit-Tests und Status-Tests gruen; Fixture-Startcheck ohne Traceback und mit sichtbaren Image Jobs.
- Nicht gebaut: Live-Pipeline, echte CLI-Eingabe, Job-Submit, Render, API/n8n, neue Runs oder globale Layoutaenderung.

## 2026-05-09 Textual Cockpit Stage Router V0.1
- Pipeline Map V1 ist die neue Operator-Sicht im Textual Cockpit: 00 Command Center bis 15 Final Output.
- Active Workspace routet anhand `selected_stage`; Default bleibt Stage 09, damit das bestehende Prompts/Image-Jobs-Panel beim normalen Start sichtbar bleibt.
- Stage 00-03 sind als initiale read-only Views gebaut, Stage 04-15 als funktionale Platzhalter mit Status/Purpose/Artifacts/Next Action.
- Auswahl ist per Pfeil hoch/runter und j/k stabil; Klick-Handler ist vorbereitet. Keine Live-Pipeline-Integration, keine CLI-Eingabe, kein Job-Submit, kein Render, keine API-/n8n-Aufrufe.
- Verifikation: Cockpit- und Status-Unit-Tests gruen; Fixture- und Missing-Run-Startchecks liefen ohne Traceback. `cockpit-video-smoke-001` war nicht vorhanden.

## 2026-05-08 Textual Cockpit Header / Active Workspace Reference Pass
- Header ist naeher an Referenz-/Command-Center-Struktur: links CM-Brand + `CONTENT MASCHINE LIVE`, mittig Job/Pipeline/Mode/Format/Status, rechts Time UTC/Session/Operator/Run Type/Watch.
- Active Workspace oben hat jetzt echte innere Subbereiche: `CURRENT POSITION AND PIPELINE PATH`, `PROMPTS / IMAGE JOBS`, `PIPELINE FLOW`.
- Prompt/Image-Jobs ersetzen dort die alte einfache Scene-Card-Liste; pro Zeile stehen Nummer, Scene-ID, gekuerzte Summary, Source/Keyframe und Status.
- Flow-Streifen ist statisch und read-only; keine Fake-Livechecks, keine Fake-ETA, keine Prozentwerte, kein Render und keine neuen Runs.
- Verifikation: Cockpit- und Status-Unit-Tests gruen; Fixture-Startcheck ohne Traceback; `cockpit-video-smoke-001` nicht vorhanden.

## 2026-05-08 Textual Cockpit Active Workspace Detail Pass
- Active Workspace wurde gezielt verbessert, ohne Redesign: Status-Zone ist kompakter, Stage Output aussagekraeftiger, Scene Cards bleiben im bestehenden Stil und Run Notes nennen Session/Watch/read-only.
- Creative-OS-Fixture zeigt weiterhin 3 kompakte Scene Cards; Agent-Core-Runs zeigen Scene Count, final.mp4 present/missing, director_mode, optional stop_after und gekuerzte Scene-/Prompt-Zusammenfassungen.
- Skill Health / Issues Semantik bleibt unveraendert: Runtime-/Director-/final.mp4-Probleme gehoeren nicht in Skill Health.
- Verifikation: `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v` und `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` waren gruen.
- Startcheck Fixture lief ohne Traceback; `cockpit-video-smoke-001` war in `/workspace/agent_runs` nicht vorhanden.
- Nicht gebaut: Redesign, neue Runs, Render, LTX-Ausbau, n8n, API-Integration oder Pipeline-Integration.

## 2026-05-07 Final Closeout Cockpit Semantics + Video Proof
- Cockpit-Semantik ist final verengt: Director-Fallback, Director inactive, `final.mp4 missing`, Run missing/unknown gehoeren in Issues/Next; Skill Health zeigt keine Runtime-Probleme mehr.
- Issues Tile nutzt Severity-Klassen: `issues-none`, `issues-warning`, `issues-error`; Border bleibt im bestehenden Look, nur Severity-Farbe wechselt.
- Finaler Proof-Run: `cockpit-video-smoke-001` unter `/workspace/agent_runs`, ohne `--stop-after`, `director_mode=llm_augmented`, `director_llm_active=true`, `final.mp4` vorhanden.
- Cockpit V0.4 Watch gegen den Proof-Run zeigte `FINAL MP4 ✓ present`, `agent_core`, Watch on, keine Director-Fallback-Issue und Skill Health `✓ ok`.
- Keine neue Architektur, kein Redesign, keine n8n-/Settings-/Pipeline-Integration.

## 2026-05-07 Director 8011 Restore + Cockpit Smoke
- Director 8011 ist wieder erreichbar. Fix war kein Rebuild: `ensure_llama_cpp.sh` reparierte vorhandene `llama.cpp`-Runtime, danach startete `init.sh` den `llama-server`.
- Vorher: `/v1/models` auf 8011 nicht erreichbar, kein `llama-server` Prozess; `llama-server` hatte kein Execute-Bit und lokale `.so.0`-Links fehlten.
- Nachher: `check_director_llm.py` meldet `director_llm_active=true` fuer `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`.
- Neuer Smoke-Run: `cockpit-realrun-smoke-002`, Agent-Core-Root-Artefakte vorhanden, `director_mode=llm_augmented`, `director_llm_active=true`, kein `final.mp4`, weil bewusst mit `--stop-after model_prompts` gelaufen.
- Cockpit V0.4 Watch gegen den Run bleibt read-only; kein Director-Fallback-Issue sichtbar, `final.mp4 missing` bleibt korrekt sichtbar.

## 2026-05-07 Textual Cockpit V0.4
- Cockpit-State erkennt jetzt `creative_os`, `agent_core`, `missing` und `unknown` Run-Typen.
- Agent-Core-Runs werden read-only direkt aus `/workspace/agent_runs/<job-id>` gelesen; es ist kein `creative_os/` Unterordner noetig, damit Header/Artifacts/Issues sinnvolle Daten zeigen.
- Sichtbar gemappte Agent-Core-Dateien: `result.json`, `state.json`, `plan.json`, `scene_plan.json`, `model_prompts.json`, `prompt_audit.json`, `director_output.json`, `final.mp4`.
- `rule_based_fallback`, `director_llm_active=false` und fehlendes `final.mp4` erscheinen als Issues mit Next-Hinweis auf Director 8011 bzw. unvollstaendigen Final Output.
- Creative-OS-Fixture-Start bleibt kompatibel; keine Layout-/Theme-Aenderung, keine Pipeline-Integration, kein Render, kein LTX, kein Director-Fix.

## 2026-05-07 Textual Cockpit V0.3
- Cockpit ist jetzt fuer echte Run-Artefakte read-only vorbereitet: `python3 /workspace/scripts/creative_os_cockpit.py --job-id <job-id>` nutzt defaultmaessig `/workspace/agent_runs`.
- Fixture/Demo bleibt unveraendert: `--runs-root /workspace/tests/fixtures/creative_os_runs` zeigt `Session fixture/demo`.
- Watch Mode: `--watch --refresh-sec 2` liest State/Inspector-Daten periodisch neu; `r` bleibt manueller Refresh, `q` beendet.
- Fehlende Runs zeigen einen Missing-State im Workspace statt Python-Traceback.
- `/workspace/agent_runs` bleibt fluechtig und nur Datenquelle; das Cockpit schreibt dort nichts und integriert keine Pipeline-/Render-Aktionen.
- Naechste sinnvolle Phase: Real-Run-Watch mit einem aktuellen echten Run visuell pruefen und danach gezielt entscheiden, welche Run-Artefakte zusaetzlich angezeigt werden sollen.

## 2026-05-07 Textual Cockpit Panel V0.2
- Textual Cockpit ist intern modularisiert unter `agent_core/creative_os/cockpit/`.
- Hauptmodule: `app.py`, `theme.py`, `layout.py`, `panel_registry.py`, `panel_types.py`, `state_adapter.py`, plus `panels/` fuer Header, System Status, Pipeline Map, Active Workspace, Skill Health, Artifacts, Issues und Next.
- `agent_core/creative_os/textual_cockpit.py` bleibt als Kompatibilitaetsimport fuer Script/Tests erhalten.
- Sichtbarer Look wurde bewusst beibehalten; keine echte Pipeline-Integration, kein Stage 8, kein Render, kein LTX, kein Video, keine API und kein n8n.
- Naechste sinnvolle Phase: Real-Run-/Watch-/Integration entwerfen, ohne die Panel-Registry wieder hart mit Pipeline-Ausfuehrung zu koppeln.

## 2026-05-06 Creative OS Textual Cockpit V0.1
- Textual-TUI-Prototyp ist gebaut: `scripts/creative_os_cockpit.py`.
- Startbefehl: `python3 /workspace/scripts/creative_os_cockpit.py --job-id creative-os-jungle-001 --runs-root /workspace/tests/fixtures/creative_os_runs`.
- Bedienung: `q` beendet, `r` liest Fixture-Daten neu ein, `h` blendet Hilfe/Keybinds ein.
- Die App nutzt `CreativeOSRunInspector` und schreibt keine Artefakte. `/workspace/agent_runs` bleibt fluechtig und unberuehrt.
- Status-CLI bleibt erhalten: `scripts/creative_os_status.py` plain/rich wurde nicht ersetzt.
- Falls Textual in einer frischen Umgebung fehlt: `python3 -m pip install 'textual>=0.89,<1.0'` oder Requirements installieren.
- Nicht gebaut/gestartet: Stage 8, Render, LTX, Video, API, n8n, Backend-Aufrufe oder neue Creative-OS-Stages.

## 2026-05-06 Creative OS Rich Cockpit V1.7
- Rich-Cockpit V1.7 ist umgesetzt in `agent_core/creative_os/dashboard.py`.
- Fuer Design, Tests und Snapshots immer explizit nutzen: `--runs-root /workspace/tests/fixtures/creative_os_runs`.
- Header zeigt bei Fixture-Root `Session fixture/demo`; dadurch entsteht kein falscher Eindruck eines echten `/workspace/agent_runs`-Runs.
- Snapshots liegen unter `/workspace/cli_cockpit_snapshots/`; `/workspace/agent_runs` wurde nicht als Quelle verwendet und nicht mutiert.
- Nicht gebaut/gestartet: Stage 8, Render, LTX, Video, API, n8n, Textual, Backend-Livechecks.

## 2026-05-06 Creative OS CLI Cockpit
## Stand 2026-05-13 Creative OS Phase-1-Reality-Fix
- Phase 1 ist lokal bis Stage `09` verdrahtet und das Cockpit liest echte Run-Artefakte statt starrer Demo-Werte, sobald ein Real-Run vorhanden ist.
- Beispielrun: `/workspace/agent_runs/phase1-build-smoke-20260513/creative_os`.
- CLI:
  `python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1 --job-id phase1-build-smoke-20260513 --topic "jungle safari at sunrise" --pipeline shortform_storyboard_v1 --mode visual_adventure --style cinematic_nature --format portrait --duration 9s --scenes 3`
- `phase1_status.json` ist konsistent fuer fertige Stage `09`: `completed_stages` enthaelt `09`, `real_run_stage=09`, `last_completed_stage=09`, `next_available_stage=none_phase1_complete`.
- Stage `09` liest `keyframe_manifest.json` inklusive Backend, Jobstatus, Progress, Elapsed, Output-Pfad, Error, Backend-Job-ID und Datei-Metadaten.
- Fertige Jobs ohne Output-Datei werden als Error/Warn angezeigt; keine Fake-Erfolge.
- Bei echten PNGs wird `keyframe_gallery.html` erzeugt; Terminal-Cockpit zeigt echte Preview-Pfade statt Fake-Thumbnails.
- Status-CLI ist fuer Phase-1-Runs 00-09-aware und markiert Stage `10+` als nicht gebaut.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v` und `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` sind gruen.
- Nicht gebaut: Stage `10-15`, LTX-Video, Assembly, Final Output, n8n/API, neue Dependencies.

- Rich-Cockpit-Design-Pass ist umgesetzt in `agent_core/creative_os/dashboard.py`; `scripts/creative_os_status.py` bleibt der Entry.
- `scripts/creative_os_status.py` unterstuetzt `--runs-root`; `/workspace/agent_runs` ist nur Default fuer fluechtige echte Run-Artefakte, keine Systemquelle, kein Config-Ort, kein Skill-Ort und keine Pflichtabhaengigkeit.
- Tests gruen: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`.
- Snapshots: `/workspace/cli_cockpit_snapshots/overview_rich_cli.txt`, `/workspace/cli_cockpit_snapshots/all_rich_cli.txt`, `/workspace/cli_cockpit_snapshots/overview_plain_cli.txt`.
- Wichtiger Zustand: Der echte Run-Ordner `/workspace/agent_runs/creative-os-jungle-001/creative_os` kann nach Pod-Reset fehlen. Snapshots/Tests nutzen deshalb die stabile isolierte Fixture `/workspace/tests/fixtures/creative_os_runs`; echte Runs muessen bei leerem `agent_runs` neu erzeugt werden.
- Fehlende Runs sind kein Systemfehler; das CLI meldet den gesuchten Pfad und verweist auf `--runs-root` oder das Erzeugen eines echten Runs.
- Nicht gebaut/gestartet: Stage 8, Render, LTX, Video, API, n8n, Textual, Backend-Livechecks.

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
