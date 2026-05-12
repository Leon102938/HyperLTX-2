# Task Board

Stand: 2026-05-12, nach finalem Panel-Polish Stage 00-08.

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

## In Progress

- [ ] Kein aktiver Build-Durchlauf. Panel-Polish ist abgeschlossen.

## Next Build Backlog

- [ ] Stage `00` bis `08` in einem echten Terminal visuell durchklicken und nur konkrete Text-/Spacing-Abweichungen dokumentieren.
- [ ] Echte Scene-Contract-Felder aus realen Runs breiter mappen, falls neue Artefakte verfuegbar sind.
- [ ] Prompt Audit und Artifact Policy besser aus echten Artefakten lesen, falls `prompt_audit.json` verfuegbar wird.

## Later Backlog

- [ ] Entscheidung treffen, ob Referenz-Unterstufen wie `08.1` bis `08.4` nur visuell oder als echte Stage-IDs abgebildet werden sollen.
- [ ] Modellrollen und Mapping nur dann weiter ausbauen, wenn echte Datenquellen feststehen.

## Blocked / Needs Decision

- [ ] Soll die Zielstruktur `01-09` exakt als Stage IDs umgesetzt werden oder weiter als fachliche Checkliste fuer die bestehende `00-15` Registry gelten?
- [ ] Welche Artefakte gelten langfristig als Quelle fuer Modellrollen, Prompt Audit und Artifact Policy?
