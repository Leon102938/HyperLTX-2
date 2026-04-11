# CHANGELOG.md

## 2026-04-11 Bootstrap-Recon
- kanonisches Projektgedaechtnis unter `/workspace/codex` angelegt
- vorhandenen Legacy-Memory-Stand aus `/workspace/Codex` gesichtet und konsolidiert
- RunPod-Umgebung, laufende Dienste, Modellbestaende, Python-Runtimes und lokale Backends verifiziert
- `SYSTEM_AUDIT.md` als technische Bestandsaufnahme erstellt
- `COMMAND_PROMPTS.md` fuer wiederverwendbare Arbeitsbefehle und Prompt-Bausteine erstellt
- Projektstatus, aktiver Plan, Aufgabenboard, Memory und Entscheidungen auf den echten Ist-Zustand aktualisiert

## 2026-04-11 Phase-1-Core-Build
- neues Paket `agent_core/` fuer den modularen Agent-Core angelegt
- zentrale Kernbausteine implementiert: Agent, Schemas, Planner, State-Store, Backend-Registry, Assembler, Utils
- produktive Phase-1-Adapter fuer Qwen TTS und LTX2 ueber lokale FastAPI-Endpunkte gebaut
- Future-ready-Stubs fuer Music und Storyboard angelegt
- Beispieljob `examples/minimal_job.json` erstellt
- Tests `test_core_smoke.py` und `test_planner_rules.py` erstellt und erfolgreich ausgefuehrt
- Projektgedaechtnis auf den realen Implementierungsstand aktualisiert

## 2026-04-11 Real-Backend-Validation
- echter End-to-End-Core-Lauf gegen reale Qwen-TTS- und LTX2-Backends durchgefuehrt
- Qwen-TTS-Adapter real verifiziert: WAV-Output, Dauerprobe und Artefaktfluss funktionieren
- erster realer LTX2-Fehler verifiziert: Phase-1-Custom-Aufloesungen muessen Vielfache von 64 sein
- zweiter realer LTX2-Fehler verifiziert: `a2vid` mit generierter TTS-Audio war im aktuellen Setup nicht vertragstabil
- Core-Fixes umgesetzt:
  - Custom-Resolution-Validierung auf Vielfache von 64
  - Framezahl auf LTX-Schema `8k+1` geschnappt
  - Step-Details nach Re-Planung korrekt aktualisiert
  - Log-Artefakt korrekt als vorhanden markiert
  - Failure-Resultate behalten nun vorhandene Voice-Artefakte
  - Phase-1-Renderpfad auf stabilen `ti2vid`-Vertrag umgestellt
- verifizierter Erfolgs-Run `real-e2e-check-3` erzeugte echtes MP4 und echte WAV-Artefakte

## 2026-04-11 Final-MP4-Assembly
- `ResultAssembler` von reiner Referenzsammlung auf echte Final-Assembly erweitert
- neues finales Artefakt `final_output_mp4` eingefuehrt
- `ResultSummary` um `output_final_path` erweitert
- Assembler ersetzt den Audio-Stream des gerenderten LTX2-MP4 kontrolliert durch die erzeugte Qwen-TTS-Voice
- Muxing erfolgt per `ffmpeg` mit gepaddeter oder gekuerzter Voice-Spur auf Video-Laenge
- Fallback umgesetzt: ohne nutzbares Voice-Artefakt wird das Render-MP4 als `final.mp4` gespiegelt
- Smoke-Tests auf gueltige Testmedien umgestellt und um No-Voice-Fall erweitert
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 6 Tests gruen
- echter End-to-End-Mux-Lauf `real-e2e-mux-2` erfolgreich verifiziert

## 2026-04-11 Duration-Contract-Hardening
- Planner quantisiert die Ziel-Dauer jetzt einmalig und schreibt die kanonische Framezahl in den Video-Step
- LTX2-Adapter uebernimmt die geplante Framezahl jetzt direkt statt sie aus der gerundeten Plan-Dauer neu zu berechnen
- `probe_media_duration` auf `0.001s`-Praezision angehoben
- `ResultSummary` um `actual_video_duration_sec` und `actual_final_duration_sec` erweitert
- Video- und Final-Artefakte dokumentieren jetzt geplante und reale Dauerwerte explizit
- neue Tests fuer Quantisierungsstabilitaet und Assembler-Randfaelle hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 10 Tests gruen
- realer Dauervertragslauf `real-duration-case-a` verifiziert: Plan `5.041s`, Video `5.042s`, Final `5.042s`
- realer No-Voice-Lauf `real-duration-case-c` verifiziert: Plan `4.041s`, Video `4.042s`, Final `4.042s`
- realer Long-Voice-Trim-Fall `real-duration-case-b` auf Assembler-Ebene mit echten Qwen- und LTX2-Artefakten verifiziert

## 2026-04-11 Phase-2A Scene-Shot-Planning
- `ScenePlan` und `ShotPlan` in die Schemas aufgenommen
- Planner segmentiert Jobs jetzt regelbasiert in mehrere Szenen mit Dauer-, Narrations- und Prompt-Zuordnung
- jede Szene erhaelt in Phase 2A genau einen ersten renderbaren Shot als minimalen strukturierten Produktionsvertrag
- `scene_plan.json` wird pro Job als neues Artefakt gespeichert
- Agent rendert bei Multi-Segment-Jobs mehrere LTX2-Szenen nacheinander
- Assembler concateniert mehrere Rohclips zu `assembled_video.mp4` und finalisiert danach wie gewohnt zu `final.mp4`
- Single-Flow bleibt ueber `single_scene`-Fallback intakt
- neue Tests fuer Segmentierung, Dauerverteilung und Fallback hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 14 Tests gruen
- echter Multi-Segment-Lauf `real-phase2a-multiscene-1` erfolgreich verifiziert

## 2026-04-11 Phase-2B Multi-Take-Selection
- `TakePlan` und `TakeResultRecord` in die Schemas aufgenommen
- Planner plant jetzt mehrere Takes pro Szene inklusive deterministischer Seeds
- LTX2-Adapter uebergibt den geplanten Take-Seed an das reale Backend
- Agent rendert jetzt mehrere Takes pro Szene, spiegelt erfolgreiche Take-Videos in den Job-Workspace und dokumentiert pro Szene den `selected_take`
- neue Artefakte eingefuehrt:
  - `takes.json`
  - gespiegelt abgelegte Take-Videos unter `scenes/<scene_id>/takes/`
- Auswahlregel `first_successful_take` implementiert und Assembler auf selektierte Takes umgestellt
- neue Tests fuer Mehrfach-Takes, Auswahl, Fehler-Fallback und Persistenz hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 18 Tests gruen
- echter Multi-Take-Lauf `real-phase2b-multitake-1` erfolgreich verifiziert

## 2026-04-11 Tagesabschluss
- kanonisches Projektgedaechtnis unter `/workspace/codex` auf den echten Phase-2B-Endstand geschaerft
- `HANDOFF.md` fuer die naechste Session angelegt
- `.gitignore` vorsichtig um Laufzeit-/Artefaktordner, lokale Logs, Checkpoints, Legacy-Ordner und Egg-Info erweitert
- aktueller Tagesendstand erneut per `python -m unittest discover -s /workspace/tests -v` verifiziert -> 18 Tests gruen
