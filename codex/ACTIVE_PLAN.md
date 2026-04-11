# ACTIVE_PLAN.md

## Aktiver Gesamtplan
Phase 1: Agent-Core-Kern
Phase 2A: Scene-/Shot-Planung
Phase 2B: Mehrfach-Takes und Auswahl pro Szene
Phase 3: optionale API-Schicht fuer den Core
Phase 4: n8n-Anbindung
Phase 5: Ausbau, Optimierung, Spezialisierung

## Phase 2B Ziel
Den abgeschlossenen Phase-2A-Multi-Scene-Flow um Mehrfach-Takes und eine stabile Auswahl pro Szene erweitern, ohne den bestehenden Single-Flow oder die Kernvertraege zu zerlegen.

## Phase 2B Arbeitsplan
1. Take-Datenstruktur in den bestehenden Scene-Vertrag integrieren.
2. Mehrere Takes pro Szene rendern und im Job-Workspace spiegeln.
3. Erste stabile Auswahlregel `first_successful_take` implementieren.
4. `takes.json`, State-Persistenz und Assembly auf selektierte Takes erweitern.
5. Einen echten Multi-Take-Lauf verifizieren.

## Empfohlener Minimal-Vertical-Slice
- Eingabe: einfacher Job mit `prompt`, optional `tts`, optional `audio`, optional `video`
- Ablauf:
  1. Job validieren
  2. Plan erzeugen
  3. Step-State persistieren
  4. Qwen-TTS optional ausfuehren
  5. LTX-2 Step ausfuehren
  6. finales MP4 assembliert speichern

## Phase-2B-Ist-Stand
- gebaut:
  - Core-Paket `agent_core/`
  - Job-Schema, Plan-Schema, State-Schema, Result-Schema
  - regelbasierter Planner
  - filesystem-basierter State-Store
  - Backend-Registry
  - Assembler mit Final-Mux und Rohclip-Concat
  - produktive HTTP-Adapter fuer Qwen TTS und LTX2
  - Future-ready-Stubs fuer Music und Storyboard
  - Smoke-, Planner- und Assembler-Tests
  - Scene-/Shot-Schemas
  - regelbasierte Scene-Segmentierung
  - Take-Planung pro Szene
  - Auswahlregel `first_successful_take`
  - `takes.json` als Take-Artefakt
- verifiziert:
  - Tests laufen gruen
  - Voice-Dauer beeinflusst die geplante Videolaenge
  - pro Job werden Input, Plan, State, Result und Logs gespeichert
  - echter Qwen-TTS-Lauf erfolgreich
  - echter LTX2-Lauf erfolgreich
  - realer End-to-End-Core-Lauf erfolgreich mit `real-e2e-check-3`
  - realer End-to-End-Mux-Lauf erfolgreich mit `real-e2e-mux-2`
  - quantisierter Dauervertrag zwischen Planner, LTX2 und Assembler ist geschaerft
  - Phase-1-stabiler LTX2-Pfad ist aktuell `ti2vid`
  - Framezahl wird auf das reale LTX-Schema `8k+1` geschnappt
  - Custom-Aufloesungen werden fuer Phase 1 auf Vielfache von 64 begrenzt
  - Assembler erzeugt jetzt `final.mp4` im Job-Workspace
  - Voice wird in Phase 1 per `ffmpeg` in das finale MP4 gemuxt
  - wenn kein nutzbares Voice-Artefakt vorliegt, wird das gerenderte Video sauber als `final.mp4` gespiegelt
  - reale Dauerabweichung Plan zu Video ist fuer verifizierte Phase-1-Runs aktuell auf etwa `0.001s` geschrumpft
  - Randfaelle sind validiert:
    - Voice kuerzer als Video
    - Voice laenger als Video auf Assembler-Ebene mit realen Artefakten
    - kein Voice-Artefakt
  - Scene-Plan-Artefakt `scene_plan.json` wird geschrieben
  - Multi-Segment-Jobs koennen mehrere LTX2-Renders erzeugen, concateniert speichern und danach normal finalisieren
  - echter Multi-Segment-Lauf erfolgreich mit `real-phase2a-multiscene-1`
  - pro Szene koennen mehrere Takes geplant und gerendert werden
  - erfolgreiche Take-Videos werden im Job-Workspace unter `scenes/<scene_id>/takes/` gespiegelt
  - pro Szene wird ein `selected_take` dokumentiert
  - die finale Assembly verwendet nur selektierte Takes
  - `takes.json` wird geschrieben und `state.json` enthaelt die Take-/Selection-Details des Video-Steps
  - echter Multi-Take-Lauf erfolgreich mit `real-phase2b-multitake-1`
  - Tagesabschluss-Doku und Handoff werden in `/workspace/codex` gepflegt

## Reale Validierungsnotizen
- Fehlversuch 1:
  - `a2vid` mit `768x432` scheiterte, weil Two-Stage-LTX Aufloesungen als Vielfache von 64 braucht
- Fehlversuch 2:
  - `a2vid` mit gueltiger Aufloesung scheiterte trotzdem an Audio-/Latent-Shape-Mismatch
- Konsequenz:
  - `a2vid` ist fuer generierte TTS-Audio in Phase 1 nicht als stabiler Standard freigegeben
  - stabiler Core-Pfad nutzt Voice fuer Dauer- und Artefaktlogik, aber rendert Video via `ti2vid`

## Phase 1 Soll-Dateien
- `agent_core/__init__.py`
- `agent_core/agent.py`
- `agent_core/schemas.py`
- `agent_core/planner.py`
- `agent_core/state_store.py`
- `agent_core/backend_registry.py`
- `agent_core/assembler.py`
- `agent_core/utils.py`
- `agent_core/adapters/base.py`
- `agent_core/adapters/ltx2_adapter.py`
- `agent_core/adapters/qwen_tts_adapter.py`
- `agent_core/adapters/music_adapter.py`
- `agent_core/adapters/storyboard_adapter.py`
- `tests/test_core_smoke.py`
- `tests/test_planner_rules.py`
- `examples/minimal_job.json`

## Bewusst noch nicht bauen
- n8n-Integration
- neue HTTP-API fuer den Core
- GUI
- Multi-Agent-Swarm
- DB, Queue-Cluster, Event-Bus
- tiefe Umbauten in `LTX-2`, `ACE-Step-1.5` oder `Qwen3-TTS`
- vollwertige Musik-, Storyboard-, Hook- oder Quality-Subsysteme
- produktive `a2vid`-Freigabe ohne weitere reale Backend-Validierung
- weitere grosse Umbauten am Core nur fuer kosmetische Assembler-Erweiterungen

## Phase-1-Abschluss
- Der definierte Phase-1-Scope gilt jetzt als technisch sauber abgeschlossen:
  - Job validieren
  - quantisierten Produktionsplan bauen
  - optionale Voice erzeugen
  - stabiles LTX2-Video rendern
  - finales MP4 erzeugen
  - State, Result und Artefakte nachvollziehbar speichern

## Phase-2B-Abschlusskriterium
- Mehrfach-Takes pro Szene sind implementiert, der Single-Flow funktioniert weiter, selektierte Takes werden sauber persistiert und mindestens ein realer Multi-Take-Lauf ist erfolgreich verifiziert.

## Tagesabschluss
- Commit-wuerdig sind aktuell die Quellordner `agent_core/`, `tests/`, `examples/`, die geschraefte `.gitignore` und der kanonische Projekt-Memory unter `/workspace/codex`.
- Laufzeit- und Artefaktordner wie `agent_runs/`, `exports/`, `jobs/`, `status/`, `venvs/` und Checkpoints bleiben bewusst ausserhalb eines sauberen Code-Commits.
