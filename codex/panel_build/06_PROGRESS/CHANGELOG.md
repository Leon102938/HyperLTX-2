# Changelog

Diese Datei dokumentiert jede Session mit Datum, Umfang und Teststatus.

## 2026-05-15 - Skill Tree V1 + Stage 03-08 Real Wiring

### Geaendert

- Root-`skills/` mit `skill_manifest.json`, Mode-, Style-, Hook- und Model-Skill-Dateien angelegt.
- `agent_core/creative_os/skill_tree_v1.py` laedt Manifest und Markdown-Regeln robust; fehlende Skills werden `missing`.
- Stage `03` schreibt echte `skill_match.json` und `skill_tree.json`.
- Stage `04` bis `08` nutzen Skill-Regeln in Strategy, Beat/Hook, Judge, Scene Contracts und Prompt Compiler.
- Cockpit Stage `03` zeigt echte Skill-Gruppen; Stage `04` bis `08` zeigen `Source: skills loaded`.

### Tests / Smokes

- `python3 -m unittest /workspace/tests/test_skill_tree_v1.py -v`: gruen, 6 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 25 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 17 Tests.
- Smoke: `skill-tree-v1-smoke-20260515`; Stage `09` pausiert korrekt wegen missing Z-Image-Backend.

## 2026-05-14 - Phase 1 Live Orchestrator V3 Fix

### Geaendert

- Pipeline Map behandelt Stage `10` bis `15` fuer Phase-1-Runs als out-of-scope/pending, nie done/gruen.
- Stage `09` Active Workspace zeigt echte `keyframe_manifest.json`-/`live_status.json`-Daten sichtbar.
- `keyframe_manifest.json` wird waehrend Stage-09-Image-Generierung aktualisiert, nicht erst am Ende.
- Watch-Refresh ignoriert reine Refresh-Zeit-Aenderungen und erhaelt manuelle Stage-Auswahl.

### Tests / Smokes

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 17 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 25 Tests.
- Smokes: `live-v3-smoke-noimages-20260514`, `live-v3-smoke-images-20260514`.

## 2026-05-14 - Phase 1 Live Orchestrator V2 Fix

### Geaendert

- Stage `09` ist bei disabled/missing Backend nicht mehr `done`, obwohl `keyframe_manifest.json` existiert.
- `live_status.json` und `phase1_status.json` sind konsistent: paused Backend => completed `00` bis `08`, Stage `09=error`, next `09`.
- Pipeline Map nutzt Live-Stage-Status statt blind Artefakt-Anwesenheit.
- `--open-cockpit` gibt sichere Zwei-Terminal-Befehle aus und startet Textual nicht mehr als Background-Prozess im selben TTY.
- `--stage-delay-seconds`, `--generate-images`, `--no-generate-images` ergaenzt; `--no-images` bleibt Alias.

### Tests / Smokes

- Disabled-Smoke: `live-v2-smoke-20260514`, 0 PNGs, Stage `09=error`, kein `09 done` Event.
- Image-Smoke: `live-v2-smoke-images-20260514`, 3 PNGs, Gallery, Stage `09=done`.
- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 25 Tests.

## 2026-05-14 - Phase 1 Live Cockpit Orchestration

### Geaendert

- Neuer CLI-Befehl `scripts/agent_core_cli.py creative-os run-phase1-live`.
- Live-Run schreibt `live_status.json` und `stage_events.jsonl` waehrend Stage `00` bis `09` laufen.
- Live-State trennt `viewed_stage` von `real_run_stage` und `current_running_stage`.
- Cockpit Watch liest Live-State und startet Live-Runs bei Stage `00`.
- Real-Run Stage `09` nutzt nur echte `keyframe_manifest.json`-Werte; fehlendes Manifest zeigt keine Fake-Cards und Progress wird nicht geraten.
- Live-Smoke: `/workspace/agent_runs/live-smoke-20260514/creative_os`.

### Nicht geaendert

- `creative-os run-phase1` bleibt Batch-kompatibel.
- Keine Stage-10-bis-15-Runtime, kein LTX, kein n8n/API, kein Redesign, keine neue Textual-Version.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 23 Tests.

## 2026-05-13 - Phase 1 Hardening Stage 09 Retry/Resume

### Geaendert

- Frischer E2E-Run `phase1-hardening-smoke-20260513` erzeugt Artefakte `00` bis `09`, 3 echte Keyframe-PNGs und `keyframe_gallery.html`.
- Neuer CLI-Befehl `scripts/agent_core_cli.py creative-os retry-keyframes`.
- Retry/Resume liest nur `keyframe_manifest.json`, erkennt `failed/error/queued/running`, fehlende `output_path` und `file_exists=false`.
- Retry schreibt nur `keyframe_manifest.json`, `phase1_status.json` und ggf. `keyframe_gallery.html`; Stage `00` bis `08` bleiben unveraendert.
- `--dry-run`, `--scene scene_02` und `--force` sind unterstuetzt.
- Ohne `--force` werden fertige vorhandene PNGs nicht neu erzeugt.
- Cockpit Stage `09` zeigt bei fehlendem Real-Run-Manifest `missing manifest` statt Fake-Cards.
- Progress faellt nicht mehr auf harte `100%` zurueck, wenn ein Real-Run keine passende Manifest-/Dateibasis hat.

### Nicht geaendert

- Keine Stage-10-bis-15-Runtime.
- Keine LTX-Video-Generation, kein Assembly, kein Final Output.
- Keine n8n/API-Integration.
- Kein Redesign und keine neuen Dependencies.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 21 Tests.
- Smoke: `retry-keyframes --scene scene_02` hat auf `phase1-hardening-retry-sim-20260513` eine fehlende PNG neu erzeugt.

## 2026-05-13 - Phase 1 Runtime bis Stage 09

### Geaendert

- Neuer lokaler Phase-1-Runner `agent_core.creative_os.phase1_runtime`.
- `scripts/agent_core_cli.py creative-os run-phase1` schreibt Creative-OS-Artefakte unter `/workspace/agent_runs/<job-id>/creative_os/`.
- Erzeugte Artefakte: `normalized_job.json`, `pipeline_route.json`, `intent_route.json`, `mode_style.json`, `creative_direction.json`, `skill_match.json`, `skill_tree.json`, `creative_strategy.json`, `beat_hook_plan.json`, `selected_beat_plan.json`, `creative_judge.json`, `stage6_review_decision.json`, `scene_contracts.json`, `keyframe_contracts.json`, `prompt_payload_compiled.json`, `zimage_prompts.json`, `keyframe_manifest.json`, `phase1_status.json`.
- Stage `09` prueft Z-Image ueber den vorhandenen lokalen HTTP-Backendpfad und schreibt pro Szene Jobstatus `queued/running/finished/error`.
- Wenn das Image-Backend fehlt, bleibt der Run sauber auf `phase1_paused_missing_image_backend`; keine Fake-PNGs und kein Fake-Erfolg.
- Inspector und Cockpit-State lesen Phase-1-Artefakte und Stage-09-Jobs aus `keyframe_manifest.json`.
- Tests fuer Phase-1-CLI, Artefakte, Missing-Backend-Verhalten und Textual `0.89.x` ergaenzt.

### Nicht geaendert

- Keine Stage-10-bis-15-Runtime.
- Keine LTX-Video-Generation, kein Assembly, kein Final Output.
- Keine n8n/API-Integration.
- Keine neuen Dependencies und keine Textual-8.x-Anpassung.
- Kein Design-Redesign der Panels.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 15 Tests.

## 2026-05-12 - Finaler Panel-Polish 00-08

### Geaendert

- Textual-Stand bestaetigt: Runtime `0.89.1`, Pflicht-Pin `textual>=0.89,<1.0`.
- Stage `00 Command Center` stabil beibehalten; keine CLI-Eingabe gebaut.
- Stage `01 Pipeline wählen` korrigiert: Current-Position-Block entfernt, Pipeline Purpose/Overview, Pipeline Flow, Pipeline Assets und Output/Next ergaenzt.
- Stage `02 Mode & Style` erweitert: Mode Intent, Content Logic, Style Language, Visual Rules, Risks/Avoids und Handoff.
- Stage `03 Skills laden` korrigiert: sichtbarer Skill Tree, Skill Loading Progress, Loading Status, Skill Sources und Health; Pipeline bleibt technische Route und wird nicht als Skill behandelt.
- Stage `04 Creative Strategy` erweitert: A/B/C/D-Struktur mit Strategy Engine, Input Context, Skill Stack, JSON Preview und Output Summary.
- Stage `05 Beat / Hook Planner` erweitert: Hook Brief, Hook Candidates, Selected Beat Plan und Output Preview; keine Fake-Bilder.
- Stage `06 Creative Judge` erweitert: Judge Input, Creative Checks, Final Decision, Output Preview, Risiken/Fixes/Handoff.
- Stage `07 Scene Contracts` verdichtet: pro Szene Status und Kernfelder sichtbarer; Output Preview und Readiness verbessert.
- Stage `08 Prompt Compiler` stabil gehalten: geschlossener Image-Compiler-Hauptkasten, keine inneren kaputten Subboxen, keine ASCII-Pipes, rechte Compiler-Spalte erhalten.
- Tests fuer Stage-01-bis-08-Polish und Regressionspunkte aktualisiert.

### Nicht geaendert

- Keine neuen Dependencies.
- Keine Textual-8.x-Anpassung.
- Keine Pipeline-, Render-, API- oder n8n-Integration.
- Keine echten Ausfuehrungen.
- Keine Scroll-, Terminal-, Quit- oder Performance-Fixes angefasst.
- Header, Sidebar, System Status und Bottom Panels nicht redesignt.
- Stage `09` nicht umgebaut.
- Keine Flow-/Symbol-Leiste im Active Workspace zurueckgebracht.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.
- Fixture-Smoke ohne Watch gestartet; danach keine verbleibenden `creative_os_cockpit.py` Prozesse.

## 2026-05-12 - Stage 08 Closeout und Stage 01-06 Polish

### Geaendert

- Textual-Stand dokumentiert: Runtime `0.89.1`, Pflicht-Pin `textual>=0.89,<1.0`.
- Stage `08 Prompt Compiler` stabilisiert.
- `IMAGE COMPILER (ACTIVE)` wieder als geschlossener Hauptkasten gebaut.
- Innerhalb des Image-Compiler-Kastens nur einfache Textsektionen fuer Scene Contract Inputs, Scene Prompt Summaries und Final Prompt Payload belassen.
- Kaputte innere Rahmen, ASCII-Pipes, Readiness-Bloecke, A/B/C-Labels, Model Rules und Artifact Policy aus Stage `08` ferngehalten.
- Rechte Compiler-Spalte mit `VIDEO`, `AUDIO` und `MUSIC` als geschlossene Boxen erhalten.
- Stage `01` bis `06` anhand der jeweiligen Referenzen gepolished:
  - Stage `01` Pipeline Overview.
  - Stage `02` Mode & Style.
  - Stage `03` Skill Loading.
  - Stage `04` Creative Strategy.
  - Stage `05` Beat / Hook Planner ohne Fake-Bilder.
  - Stage `06` Creative Judge.
- Box-Plain-Renderer bewahrt Einrueckungen innerhalb stabiler Boxen.
- Tests fuer Stage-08-Hauptkasten und Stage-01-bis-06-Struktur aktualisiert.

### Nicht geaendert

- Keine neuen Dependencies.
- Keine Textual-8.x-Anpassung.
- Keine Pipeline-, Render-, API- oder n8n-Integration.
- Keine Scroll-, Quit-, Terminal- oder Performance-Fixes angefasst.
- Header, Sidebar, System Status, Pipeline Map global und Bottom Panels nicht redesignt.
- Stage `07` und Stage `09` nicht weiter umgebaut.
- Keine untere Flow-/Symbol-Leiste in den Active Workspace zurueckgebracht.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.
- Fixture-Smoke ohne Watch gestartet; danach keine verbleibenden `creative_os_cockpit.py` Prozesse.

## 2026-05-11 - Active Workspace Scroll Pass

### Geaendert

- Active Workspace als generischen scrollbaren Stage-Host umgesetzt.
- `#workspace` bleibt der aeussere Active-Workspace-Rahmen und ist jetzt ein `ScrollableContainer`.
- Stage-Inhalte werden in `#workspace-content` gerendert und aktualisiert.
- `panel_update_targets()` aktualisiert fuer Active Workspace gezielt `#workspace-content`.
- Scroll-Host ist nicht fokussierbar, damit `down/up`, `j/k` und `enter/space` weiter die bestehenden Cockpit-Actions steuern.
- CSS fuer `#workspace` auf vertikales Scrollen und verborgenes horizontales Overflow angepasst.
- Test ergaenzt, dass `#workspace` ein `ScrollableContainer` ist und `#workspace-content` den Stage-Inhalt haelt.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Keine Pipeline-, Render-, API- oder n8n-Integration.
- Keine inhaltlichen Aenderungen an Header, System Status, Pipeline Map oder Bottom Panels.
- Keine Stage-08- oder Stage-09-Inhaltslogik geaendert.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 19 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.

## 2026-05-11 - Stage 07 Layout Fix

### Geaendert

- Stage `07 Scene Contracts` Layout nach dem Visual-Correction-Pass stabilisiert.
- `CURRENT POSITION AND PIPELINE PATH` auf zwei kompakte Reihen mit kurzen Labels umgestellt.
- A/B/C-Hauptbereiche als stabile fixe 3-Spalten-Boxen gerendert.
- Box-Helper fuer Stage `07` so korrigiert, dass Header-, Body- und Footer-Zeilen dieselbe Breite haben.
- Lange Werte in Stage `07` werden vor der Border gekuerzt.
- B-Block bleibt die breite mittlere Scene-Contracts-Spalte; A bleibt schmal, C mittelbreit.
- System Status und Pipeline Map minimal verdichtet, ohne deren Inhalte oder Logik zu aendern.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Kein Header- oder Bottom-Panel-Umbau.
- Keine Stage-08- oder Stage-09-Logik geaendert.
- Keine Render-, API-, n8n- oder Pipeline-Integration.
- Kein neuer Content-Fokus fuer Stage `07`.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 19 Tests.

## 2026-05-11 - Stage 07 Visual Correction

### Geaendert

- Stage `07 Scene Contracts` anhand `panel_07_scene_contracts/reference.png` und `visual_analysis_verified.md` korrigiert.
- Generisches Stage-Detailpanel durch eine eigenstaendige Scene-Contracts-Struktur ersetzt.
- `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS` ergaenzt.
- `CURRENT POSITION AND PIPELINE PATH` ergaenzt.
- `A) CONTRACT INPUTS` mit Creative Strategy, Beat/Hook, Creative Judge, Mode/Style, Risk Policy und Artifact Policy ergaenzt.
- `B) SCENE CONTRACTS` mit drei kompakten Contract-Cards fuer `scene_01`, `scene_02`, `scene_03` ergaenzt.
- `C) OUTPUT PREVIEW / READINESS` mit kurzer `scene_contracts.json` Preview und Stage-08-Handoff ergaenzt.
- `HANDOFF PATH` von Creative Judge zu Image Prompt Compiler ergaenzt.
- Tests fuer Stage `07` und den Erhalt von Stage `08`/`09` aktualisiert.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Kein Header-, System-Status-, Bottom-Panel- oder Pipeline-Map-Umbau.
- Keine Render-, API- oder n8n-Integration.
- Keine finalen Image Prompts in Stage `07`.
- Keine echten Bilder, Image Generation Cards oder Compiler-Branches in Stage `07`.
- Keine neue Library.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 19 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.

## 2026-05-11 - Stage 08 Visual Correction

### Geaendert

- Stage `08 Image Prompt Compiler` anhand `panel_08_prompt_compiler/reference.png` und `visual_analysis.md` korrigiert.
- Generische `COMPILER READINESS`/`IMAGE PROMPT CARDS` Struktur durch bildnaehere Prompt-Compiler-Struktur ersetzt.
- `ACTIVE WORKSPACE - PROMPT COMPILER` ergaenzt.
- `CURRENT POSITION` mit PromptCompiler, Active Branch, Output Ready und Stage-09-Handoff ergaenzt.
- `COMPILER SCOPE / OVERVIEW` ergaenzt.
- `IMAGE COMPILER (ACTIVE)` als dominante aktive Zone mit gruenem Border ergaenzt.
- `SCENE CONTRACT INPUTS`, `SCENE PROMPT SUMMARIES`, `FINAL PROMPT PAYLOAD (JSON PREVIEW)` und `MODEL RULES / ARTIFACT POLICY` innerhalb der Image-Compiler-Zone strukturiert.
- `COMPILER FAMILY / BRANCHES` auf queued/later/optional Kontext fuer Video, Audio und Music angepasst.
- Stage-08-Tests auf die bildnaehere Struktur aktualisiert.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Kein Header-, System-Status-, Bottom-Panel- oder Pipeline-Map-Umbau.
- Keine Render-, API- oder n8n-Integration.
- Keine echten Bilder in Stage 08.
- Keine neue Library.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 18 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.

## 2026-05-11 - Visual Reference Audit und Sortierung

### Geaendert

- Neun vorhandene Pipeline-Panel-Referenzbilder unter `/workspace/codex/panel_build/Pipeline vorlage/` gefunden.
- Bilder in die kanonische Struktur unter `/workspace/codex/panel_build/01_VISUAL_REFERENCES/pipeline_panels/` kopiert.
- `REFERENCE_INDEX.md` erstellt.
- Fuer Panel `01` bis `09` jeweils eine `visual_analysis.md` anhand des sichtbaren Bildes erstellt.
- `VISUAL_SPEC.md` und `IMAGE_INSTRUCTIONS.md` um den kanonischen Referenzpfad und die Pflicht zum Lesen der Analysen erweitert.

### Nicht geaendert

- Kein UI-Code geaendert.
- Keine Tests geaendert.
- Keine Panels weitergebaut.
- Originalbilder nicht verschoben oder geloescht.

### Tests

- Keine Tests ausgefuehrt, da nur Visual-Reference-Dateien und Panel-Build-Doku aktualisiert wurden.

## 2026-05-13 - Phase 1 Reality Fix

### Geaendert

- `phase1_status.json` wird fuer fertige Stage `09` konsistent geschrieben: `completed_stages` enthaelt `09`, `real_run_stage=09`, `last_completed_stage=09`, `next_available_stage=none_phase1_complete`.
- Cockpit-State liest Phase-1-Run-Fortschritt getrennt vom aktuell ausgewaehlten Panel und zeigt bei fertiger Phase 1 `Stage 10+ not built yet`.
- Stage `08` liest Prompt-Summaries aus echten Prompt-Artefakten und markiert fehlende Werte als `missing`/`not_checked`.
- Stage `09` liest echte Jobs aus `keyframe_manifest.json`: Backend, Backend-Status, Overall-Status, Scene-ID, Status, Progress, Elapsed, Output-Pfad, Error und Backend-Job-ID.
- Stage `09` prueft pro Job, ob `output_path` existiert, und zeigt Dateigroesse/mtime an. Fertige Jobs ohne Datei werden als Error/Warn behandelt.
- Terminal-Cockpit zeigt echte Preview-Pfade statt Fake-Thumbnails.
- Runtime erzeugt bei vorhandenen PNGs `keyframe_gallery.html`; der Smoke-Run `phase1-build-smoke-20260513` wurde damit aktualisiert.
- Status-CLI zeigt Phase-1-Runs mit Stage `00` bis `09` statt alter LTX-Stage-Legacy-Beschriftung.

### Nicht geaendert

- Keine Stage-10-bis-15-Runtime.
- Keine LTX-Video-Generation, kein Assembly, kein Final Output.
- Kein Redesign, keine neuen Dependencies, keine Textual-8.x-Aenderungen.
- Keine n8n/API-Integration.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 17 Tests.

## 2026-05-11 - Stage 08 Compiler Pass

### Geaendert

- Stage `08 Image Prompt Compiler` im Active Workspace neu strukturiert.
- `COMPILER READINESS` ergaenzt: Pipeline/Route, Current Stage, Scene Contracts, Creative Judge, Style Lock, Artifact Policy, Model Rules.
- `COMPILER FAMILY` ergaenzt: Image Compiler aktiv; Video, Audio und Music nur als spaeterer Kontext.
- `IMAGE PROMPT CARDS` fuer Scene `01` bis `03` ergaenzt.
- `MODEL RULES` mit zimage rules, Artifact-Bans und Stage-09-Handoff ergaenzt.
- `OUTPUT / NEXT` mit image prompts, prompt audit und naechster Stage ergaenzt.
- Test `test_stage08_image_prompt_compiler_workspace` ergaenzt.
- Bestehender Stage-Router-Test fuer die neue eigenstaendige Stage-08-Struktur angepasst.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Kein Header-, System-Status-, Bottom-Panel- oder Pipeline-Map-Umbau.
- Keine Render-, API- oder n8n-Integration.
- Keine echten Bilder in Stage 08.
- Keine neue Library.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 18 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.

## 2026-05-11 - Stage 09 Readiness Pass

### Geaendert

- Stage `09 Image / Keyframe Generation` im Active Workspace um `READINESS / INPUTS` erweitert.
- Readiness-Zone zeigt Pipeline/Route, Creative Inputs, Prompt Inputs, Model/Backend und Keyframe Readiness.
- Readiness-Werte werden aus vorhandenen Cockpit-State- und Artifact-Daten abgeleitet.
- Demo-/Fixture-Kontext wird sichtbar als `fixture/demo` ausgegeben.
- Fehlende Daten bleiben `missing` oder `not_checked` statt als echte Livewerte ausgegeben zu werden.
- Test `test_stage09_readiness_zone_keeps_image_cards` ergaenzt.

### Nicht geaendert

- Keine Stage-ID-Neuordnung.
- Kein Header-, System-Status-, Bottom-Panel- oder Pipeline-Map-Umbau.
- Keine Render-, API- oder n8n-Integration.
- Keine neue Library.

### Tests

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 17 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.

## 2026-05-11 - Setup

### Geaendert

- Aufgabenstruktur unter `/workspace/codex/panel_build` angelegt.
- Startinhalte fuer Read-First, Visual References, Design System, Panel Specs, Behavior, Testing und Progress erstellt.

### Nicht geaendert

- Kein UI-Code geaendert.
- Keine Panels implementiert.
- Keine App-Struktur umgebaut.

### Tests

- Keine Build- oder UI-Tests ausgefuehrt, da nur Markdown-Spezifikationsdateien angelegt wurden.
