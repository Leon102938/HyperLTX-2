# Active Plan

Stand: 2026-05-12, finaler Panel-Polish Stage 00-08 abgeschlossen.

## Aktueller Build-Stand

Textual bleibt fest auf der stabilen 0.89.x-Linie. Im Pod ist `textual==0.89.1` aktiv, und der Pin `textual>=0.89,<1.0` bleibt Pflicht. Keine neuen Dependency-Experimente und keine Textual-8.x-Anpassungen.

Der heutige Abschluss-Polish hat Stage `00` bis `08` im Active Workspace als stabile Content-Maschine-Panels nachgeschaerft:

- klare geschlossene Panels
- bessere Platznutzung
- mehr innere Struktur
- keine ASCII-Pipe-Trenner
- keine offenen Rahmen
- lange Texte werden gekuerzt
- Stage Map bleibt konsistent zur gewaehlten Stage

## Stage 00-08 Status

- Stage `00 Command Center`: stabiler Kontroll-/Statusbereich beibehalten; keine CLI-Eingabe gebaut.
- Stage `01 Pipeline wählen`: Current-Position-Block entfernt; Pipeline Purpose/Overview, Pipeline Flow, Pipeline Assets und Output/Next sichtbar.
- Stage `02 Mode & Style`: bestehende Struktur erhalten und mit Mode Intent, Style Language, Visual Rules, Risks/Avoids und Handoff ergaenzt.
- Stage `03 Skills laden`: Skill Tree V1 als sichtbarer Tree, Skill Loading Progress, Loading Status, Skill Sources und Health; Pipeline selbst wird nicht als Skill behandelt.
- Stage `04 Creative Strategy`: A/B/C/D-Struktur mit Strategy Engine, Input Context, Skill Stack, JSON Preview und Output Summary.
- Stage `05 Beat / Hook Planner`: Hook Brief, textbasierte Hook Candidates, Selected Beat Plan und Output Preview; keine Fake-Bilder.
- Stage `06 Creative Judge`: Judge Input, Creative Checks, Final Creative Decision, Output Preview, Risiken/Fixes/Handoff.
- Stage `07 Scene Contracts`: A/B/C-Struktur erhalten; pro Szene Status, Visual Anchor, Environment, Action, Camera/Lighting, Allowed/Forbidden Visuals und Risk sichtbarer; Readiness verbessert.
- Stage `08 Prompt Compiler`: stabiler geschlossener `IMAGE COMPILER (ACTIVE)` Hauptkasten bleibt erhalten; rechte Video/Audio/Music-Compiler-Spalte bleibt erhalten.

## Nicht Geaendert

- Stage `09` nicht umgebaut.
- Header, Sidebar, System Status und Bottom Panels nicht redesignt.
- Active-Workspace-Scroll bleibt.
- Keine Pipeline-, Render-, API- oder n8n-Integration.
- Keine echten Ausfuehrungen.
- Keine Scroll-, Terminal-, Quit- oder Performance-Fixes angefasst.
- Keine Flow-/Symbol-Leiste im Active Workspace zurueckgebracht.

## Teststatus

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests.
- Fixture-Smoke ohne Watch gestartet; danach wurden verbleibende `creative_os_cockpit.py` Prozesse beendet und geprueft.

## Naechster enger Build-Schritt

Stage `00` bis `08` in einem frischen interaktiven Terminal einmal durchklicken und nur konkrete Text-/Spacing-Abweichungen notieren.
