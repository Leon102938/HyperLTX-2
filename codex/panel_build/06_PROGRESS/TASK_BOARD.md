# Task Board

Stand: 2026-05-13, Phase-1-Hardening Stage 00-09.

## Done

- [x] Aufgabenordner `panel_build` angelegt.
- [x] Bestehende Cockpit-Dateien und Panel-Spezifikationen gelesen.
- [x] Visual References `01` bis `09` in kanonischer Struktur abgelegt und analysiert.
- [x] Textual-Drift als Hauptursache fuer Cockpit-Lag identifiziert; stabile Linie ist `textual>=0.89,<1.0`.
- [x] Stage-Navigation per Pfeiltasten und Stage-09-Image-Job-Toggle abgesichert.
- [x] Active Workspace als generischen Scroll-Host mit `#workspace-content` umgesetzt.
- [x] Stage `07 Scene Contracts` als eigenstaendige Scene-Contracts-Ansicht umgesetzt.
- [x] Stage `08 Prompt Compiler` stabilisiert: geschlossener Image-Compiler-Hauptkasten, keine kaputten inneren Rahmen, keine ASCII-Pipes.
- [x] Stage `01 Pipeline wählen` korrigiert: kein Current-Position-Block, Pipeline Purpose, Pipeline Flow, Pipeline Assets, Outputs/Next.
- [x] Stage `02 Mode & Style` erweitert: Mode Intent, Style Language, Visual Rules, Risks/Avoids, Handoff.
- [x] Stage `03 Skills laden` korrigiert: echter Skill Tree, Skill Loading Progress, Skill Sources/Health; Pipeline ist keine Skill-Gruppe.
- [x] Stage `04 Creative Strategy` erweitert: Strategy Engine, Input Context, Skill Stack, JSON Preview, Output Summary.
- [x] Stage `05 Beat / Hook Planner` erweitert: Hook Brief, Hook Candidates, Selected Beat Plan, Output Preview, keine Fake-Bilder.
- [x] Stage `06 Creative Judge` erweitert: Judge Input, Creative Checks, Final Decision, Output Preview, Risks/Fixes/Handoff.
- [x] Stage `07 Scene Contracts` verdichtet: pro Szene Status und alle Kernfelder sichtbarer, Readiness verbessert.
- [x] Stage `08` nicht kaputt gemacht; rechte Compiler-Spalte erhalten.
- [x] Tests fuer Stage `01` bis `08` Regressionspunkte aktualisiert.
- [x] Cockpit- und Status-Tests gruen.
- [x] Phase-1-CLI `creative-os run-phase1` ergaenzt.
- [x] Runtime schreibt strukturierte Artefakte fuer Stage `00` bis `09` unter `/workspace/agent_runs/<job-id>/creative_os/`.
- [x] Stage `09` erstellt `keyframe_manifest.json` mit echten Jobstatuswerten und meldet fehlendes Z-Image-Backend als `error`/paused statt Fake-Erfolg.
- [x] Cockpit-State liest echte Phase-1-Artefakte und Stage-09-Jobs aus `keyframe_manifest.json`.
- [x] Tests fuer Phase-1-Artefakte, Missing-Backend-Verhalten und Textual-0.89.x-Pin ergaenzt.
- [x] `phase1_status.json` enthaelt bei fertigem Stage `09` auch `completed_stages=["00"... "09"]`.
- [x] Cockpit unterscheidet Real-Run-Fortschritt, ausgewähltes Panel, last completed und next available.
- [x] Stage `09` zeigt Manifest-Felder, echte Preview-Pfade, Datei-Existenz, Dateigroesse und fehlende Outputs als Error/Warn.
- [x] Neue fertige Phase-1-Runs koennen `keyframe_gallery.html` aus echten PNGs erzeugen.
- [x] Status-CLI zeigt fuer Phase-1-Runs 00-09 statt alter LTX-Stage-Legacy-Beschriftung.
- [x] Frischer E2E-Hardening-Run `phase1-hardening-smoke-20260513` erzeugt Artefakte `00` bis `09`, 3 echte PNGs und Gallery.
- [x] CLI `creative-os retry-keyframes` gebaut.
- [x] Retry Dry-Run, Scene-Filter, Force-Schutz und Missing-Output-Erkennung getestet.
- [x] Retry schreibt Stage `00` bis `08` nicht neu.
- [x] Real-Run ohne `keyframe_manifest.json` zeigt keine Fake-Stage-09-Cards.

## In Progress

- [ ] Kein aktiver Build-Durchlauf. Phase 1 ist lokal gehaertet; Stage 10+ ist bewusst nicht gebaut.

## Next Build Backlog

- [ ] Stage-09-Review/Gate fuer echte Keyframes definieren, ohne Stage `10+` Runtime zu bauen.
- [ ] Echte Scene-Contract-Felder aus realen Runs breiter mappen, falls neue Artefakte verfuegbar sind.
- [ ] Prompt Audit und Artifact Policy besser aus echten Artefakten lesen, falls `prompt_audit.json` verfuegbar wird.

## Later Backlog

- [ ] Entscheidung treffen, ob Referenz-Unterstufen wie `08.1` bis `08.4` nur visuell oder als echte Stage-IDs abgebildet werden sollen.
- [ ] Modellrollen und Mapping nur dann weiter ausbauen, wenn echte Datenquellen feststehen.

## Blocked / Needs Decision

- [ ] Soll die Zielstruktur `01-09` exakt als Stage IDs umgesetzt werden oder weiter als fachliche Checkliste fuer die bestehende `00-15` Registry gelten?
- [ ] Welche Artefakte gelten langfristig als Quelle fuer Modellrollen, Prompt Audit und Artifact Policy?
- [x] Soll Stage `09` spaeter eine Resume-/Retry-Only-Image-Generation bekommen, ohne Stage `00` bis `08` neu zu schreiben? Antwort Phase 1 Hardening: ja, enger `retry-keyframes`-Befehl ist gebaut.
