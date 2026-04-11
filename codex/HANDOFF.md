# HANDOFF.md

## Aktueller Gesamtstand
- Git-Root ist `/workspace`.
- Kanonischer Projekt-Memory-Pfad ist `/workspace/codex`.
- Phase 1 ist technisch abgeschlossen:
  - Job validieren
  - Plan bauen
  - optionale Qwen-TTS-Voice erzeugen
  - stabiles LTX2-Video rendern
  - `final.mp4` erzeugen
  - State, Result und Artefakte schreiben
- Phase 2A ist abgeschlossen:
  - regelbasierte Scene-/Shot-Planung
  - Multi-Segment-Flow
  - `scene_plan.json`
- Phase 2B ist abgeschlossen:
  - mehrere Takes pro Szene
  - deterministische Seeds pro Take
  - `takes.json`
  - Auswahlregel `first_successful_take`
  - finale Assembly nur aus selektierten Takes
- Phase 2C ist abgeschlossen:
  - technischer Quality-Guard pro Take
  - `review_status` und `validation` pro Take
  - Auswahlregel `quality_guarded_best_valid_take`
  - kleine Retry-Regeln pro Szene
  - finale Assembly nur aus validierten selektierten Takes
- Phase 2D ist abgeschlossen:
  - regelbasierte Shot-/Prompt-Variation-Engine pro Szene
  - `variations` im Scene-Plan
  - Variationen sind mit Takes verknuepft
  - ausgewaehlte Variation wird pro Szene dokumentiert
- Phase 2E ist abgeschlossen:
  - kleine regelbasierte kreative Varianten-/Take-Auswahl ueber dem technischen Guard-Vertrag
  - `technical_score`, `creative_score`, `selection_reason` und `selected_by_rule` werden persistiert
  - benachbarte Szenen koennen Shot-Wiederholungen jetzt aktiv vermeiden
- Phase 3A ist abgeschlossen:
  - optionale Storyboard-/Keyframe-Pipeline ueber Z-Image
  - `storyboard_plan.json` sowie Keyframe-Kandidaten und selektierte Keyframes pro Szene
  - selektierte Keyframes werden in State, Result und Take-Metadaten gespiegelt

## Was real verifiziert wurde
- Tests:
  - `python -m unittest discover -s /workspace/tests -v` -> 33 Tests gruen
- Reale Core-/Backend-Laeufe:
  - `real-e2e-check-3`
  - `real-e2e-mux-2`
  - `real-duration-case-a`
  - `real-duration-case-b`
  - `real-duration-case-c`
  - `real-phase2a-multiscene-1`
  - `real-phase2b-multitake-1`
  - `real-phase2c-quality-guard-1`
  - `real-phase2d-variation-1`
  - `real-phase2e-creative-selection-1`
  - `real-phase3a-storyboard-1`
- Reale lokale Backends verifiziert:
  - Qwen TTS ueber vorhandene FastAPI-Endpunkte
  - LTX2 `ti2vid` ueber vorhandene FastAPI-Endpunkte
  - Z-Image ueber vorhandene FastAPI-Endpunkte

## Welche Phasen abgeschlossen sind
- Abgeschlossen:
  - Bootstrap/Recon
  - Phase 1
  - Phase 2A
  - Phase 2B
  - Phase 2C
  - Phase 2D
  - Phase 2E
  - Phase 3A
- Noch nicht gebaut:
  - externe API-Schicht
  - n8n-Anbindung
  - Musik-Pipeline
  - Hook-/Quality-Subsysteme
  - zweiter produktiver Backend-Pfad im neuen Core

## Was als Naechstes sinnvoll ist
- Kleinster sinnvoller naechster Schritt:
  - keyframe-gestuetzten Video-Pfad im bestehenden Stack vorsichtig vorbereiten
- Alternative:
  - zweiten produktiven Backend-Pfad gezielt waehlen oder spaeter kontrollierte Hook-/Narrativ-Regeln definieren
- Nicht sinnvoll als naechster Schritt:
  - neue API-Schicht
  - n8n
  - GUI
  - grosser Refactor des bestehenden Kernflusses

## Wichtige Ordner und Dateien
- Code:
  - `/workspace/agent_core`
  - `/workspace/agent_core/adapters`
- Tests:
  - `/workspace/tests`
- Beispiel:
  - `/workspace/examples/minimal_job.json`
- Kanonisches Projektgedaechtnis:
  - `/workspace/codex/PROJECT_STATE.md`
  - `/workspace/codex/ACTIVE_PLAN.md`
  - `/workspace/codex/TASK_BOARD.md`
  - `/workspace/codex/CHANGELOG.md`
  - `/workspace/codex/MEMORY.md`
  - `/workspace/codex/DECISIONS.md`
  - `/workspace/codex/COMMAND_PROMPTS.md`
  - `/workspace/codex/HANDOFF.md`
- Relevante vorhandene Backend-Huelle:
  - `/workspace/app`
- Repo-Hygiene:
  - `/workspace/.gitignore`

## Ordner, die nur Laufzeit-/Artefaktordner sind
- `/workspace/agent_runs`
- `/workspace/exports`
- `/workspace/jobs`
- `/workspace/status`
- `/workspace/venvs`
- `/workspace/LTX-2/checkpoints`
- `/workspace/ACE-Step-1.5/checkpoints`
- `/workspace/Codex`:
  - Legacy/Altbestand, nicht kanonisch

## Was commit/push-wuerdig ist
- `/workspace/agent_core`
- `/workspace/tests`
- `/workspace/examples`
- `/workspace/.gitignore`
- `/workspace/codex`

## Was eher nicht commitet werden sollte
- Laufzeit- und Artefaktordner:
  - `agent_runs/`
  - `exports/`
  - `jobs/`
  - `status/`
  - `venvs/`
- grosse Checkpoints und lokale Modelle
- lokale Pod-Logs:
  - `fastapi.log`
  - `jupyter.log`
- `__pycache__`, `.ipynb_checkpoints`, `*.egg-info`
- Legacy-Ordner `/workspace/Codex`

## Offene Risiken
- `a2vid` ist im aktuellen Setup nicht als stabiler Produktionsvertrag verifiziert.
- die Phase-2E-Auswahl ist bewusst klein und regelbasiert, aber noch keine tiefe Bildinhalts- oder Hook-Bewertung.
- Phase 3A erzeugt echte Storyboard-Keyframes, aber nutzt sie noch nicht als harten i2v-Vertrag.
- Multi-Segment-Concat kann noch kleine Timing-Deltas erzeugen.
- Der Worktree ist lokal deutlich verschmutzt durch Runtime- und Modellordner; saubere Commits muessen gezielt nur Code und Doku umfassen.
- `init.sh` ist bereits lokal modifiziert und nicht von dieser Session bereinigt worden.

## Wiederaufnahme-Prompt fuer die naechste Session
```text
Lies zuerst in /workspace/codex:
AGENTS.md, MISSION.md, USER_PREFERENCES.md, PROJECT_STATE.md, ACTIVE_PLAN.md, MEMORY.md, DECISIONS.md, CHANGELOG.md, TASK_BOARD.md, COMMAND_PROMPTS.md und HANDOFF.md.

Behandle nur /workspace/codex als kanonisches Projektgedaechtnis.
Nutze den bestehenden Phase-1-, 2A-, 2B-, 2C-, 2D-, 2E- und 3A-Stand unveraendert als Basis.
Keine n8n-Anbindung, keine externe API-Schicht, keine GUI und kein grosser Refactor, ausser der neue Auftrag verlangt das explizit.

Arbeite danach auf Basis verifizierter Fakten und aktualisiere die Memory-Dateien sauber.
```
