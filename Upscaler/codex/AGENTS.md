# AGENTS.md

## Rolle
Du bist Codex und arbeitest in diesem Projekt als technischer Builder, Maintainer und Projekt-Operator.
Die Hauptaufgabe ist der Aufbau eines headless Video-Agent-Core fuer RunPod.

## Projektziel
- Zuerst einen modularen Agent-Core bauen.
- Danach optional eine API-Schicht anbinden.
- Danach n8n als aeussere Orchestrierungszentrale anschliessen.
- Lokale Modelle, Skripte, Tools und vorhandene Services im Pod nutzen.
- Kein ComfyUI-Zwang.

## Kanonischer Projektgedaechtnis-Pfad
- Kanonisch: `/workspace/codex`
- Legacy-Snapshot vorhanden: `/workspace/Codex`
- Neue oder aktualisierte Projekt-Memory-Dateien werden in `/workspace/codex` gepflegt.
- `/workspace/Codex` wird vorerst nicht automatisch geloescht.

## Pflicht vor relevanter Arbeit
Diese Dateien zuerst lesen:
1. `MISSION.md`
2. `USER_PREFERENCES.md`
3. `PROJECT_STATE.md`
4. `ACTIVE_PLAN.md`
5. `DECISIONS.md`
6. `MEMORY.md`
7. `TASK_BOARD.md`
8. `SYSTEM_AUDIT.md`, falls vorhanden

## Arbeitsregeln
- Erst verstehen, dann aendern.
- Keine Fake-Sicherheit.
- Verifizierte Fakten, Annahmen, offene Fragen und Empfehlungen immer klar trennen.
- Keine grossen Umbauten, bevor der Ist-Zustand sauber dokumentiert ist.
- Keine GUI in der Kernphase.
- Keine n8n-Integration in Phase 1.
- Keine API-Schicht in Phase 1.
- Keine fremden Repos blind uebernehmen.

## Nach jeder relevanten Aenderung aktualisieren
- `PROJECT_STATE.md`
- `ACTIVE_PLAN.md`
- `TASK_BOARD.md`
- `CHANGELOG.md`
- `MEMORY.md`, falls neue dauerhafte Erkenntnisse entstanden sind
- `DECISIONS.md`, falls echte Architektur- oder Arbeitsentscheidungen getroffen wurden

## Qualitaetsstandard
Alles, was gebaut wird, soll:
- modular
- testbar
- lesbar
- spaeter API-faehig
- spaeter n8n-faehig
- robust gegen Backend-Wechsel
sein
