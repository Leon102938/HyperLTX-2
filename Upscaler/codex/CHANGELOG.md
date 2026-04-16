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

## 2026-04-11 Phase-2C Technical Quality Guard
- `TakeValidationReport` und `TakeRetryRecord` in die Schemas aufgenommen
- jeder erfolgreiche Take wird jetzt technisch per Dateicheck, `ffprobe`, Decode-Check, Aufloesung, FPS und plausibler Dauer validiert
- jeder Take dokumentiert jetzt `review_status` plus strukturierten `validation`-Block
- neue Auswahlregel `quality_guarded_best_valid_take` implementiert
- `first_successful_take` bleibt nur noch als Tie-Break/Fallback fuer technisch gleichwertige valide Kandidaten erhalten
- begrenzte Retry-Regeln pro Szene eingefuehrt; technisch abgelehnte Takes koennen einmalig nachgerendert werden
- `takes.json` und `state.json` dokumentieren jetzt Retry-Historie, Guard-Status und Auswahlgrund pro Szene
- Assembler bricht jetzt ab, wenn ein nicht validierter selektierter Take uebergeben wird
- neue Tests fuer Quality-Guard-Basis, Auswahl, Retry-Fallback und State-Persistenz hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 22 Tests gruen
- echter Phase-2C-Lauf `real-phase2c-quality-guard-1` erfolgreich verifiziert

## 2026-04-11 Phase-2D Shot Prompt Variation
- `VariationPlan` in die Schemas aufgenommen
- Planner erzeugt jetzt pro Szene mehrere regelbasierte kreative Varianten mit `variation_id`, `shot_type`, Kamera-Hinweis, `framing_hint`, `prompt_delta` und `prompt_variant_text`
- pro Variation koennen mehrere Takes geplant werden; der bestehende Take-Vertrag bleibt kompatibel
- Takes und Resultate dokumentieren jetzt auch ihre Quell-Variation
- `scene_plan.json`, `takes.json` und `state.json` dokumentieren jetzt Varianten, Variantenzuordnung und die ausgewaehlte Variation pro Szene
- Quality-Guard, Retry-Regeln und Assembler bleiben mit dem Variantenvertrag kompatibel
- neue Tests fuer Variations-Erzeugung, stabile Plan-Struktur, Multi-Take-Kompatibilitaet und Persistenz hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 25 Tests gruen
- echter Phase-2D-Lauf `real-phase2d-variation-1` erfolgreich verifiziert

## 2026-04-11 Phase-2E Creative Selection
- Take-Selektion um eine kleine regelbasierte kreative Heuristik ueber dem bestehenden technischen Guard-Vertrag erweitert
- kreative Auswahl beruecksichtigt jetzt Szenenposition, `shot_type`, `framing_hint`, Prompt-Variante, grobe Szenenziel-Passung und Abwechslung gegenueber benachbarten Szenen
- pro selektiertem Take und pro Szene werden jetzt `technical_score`, `creative_score`, `selection_reason` und `selected_by_rule` persistiert
- Tie-Break zwischen technisch und kreativ gleichwertigen Kandidaten faellt weiterhin kontrolliert auf `first_successful_take`
- neue Tests fuer kreative Auswahlregeln, Shot-Diversitaet benachbarter Szenen, Tie-Break und Persistenz hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 28 Tests gruen
- echter Phase-2E-Lauf `real-phase2e-creative-selection-1` erfolgreich verifiziert

## 2026-04-11 Phase-3A Storyboard Keyframes
- vorhandenen Pod-Bildpfad bewertet und Z-Image als kleinsten produktiven Storyboard-Adapter integriert
- `StoryboardConfig`, `KeyframeCandidatePlan`, `KeyframeCandidateResult`, `SelectedKeyframe` und Bildvalidierung in die Core-Schemas aufgenommen
- Planner plant jetzt optional pro Szene Storyboard-Konfiguration, priorisierte Keyframe-Kandidaten und bevorzugte Variationen
- neuer produktiver Adapter `zimage_storyboard` ueber die vorhandenen FastAPI-Endpunkte eingebunden
- `storyboard_plan.json` als neues Artefakt eingefuehrt
- Keyframe-Kandidaten werden technisch validiert, leicht selektiert und in `state.json`, `result.json` und `takes.json` dokumentiert
- der bestehende Video-Flow bleibt intakt; Storyboard-Ergebnisse werden nur als Kontext durchgereicht
- neue Tests fuer Storyboard-Planung, Persistenz, Fallback und Auswahl hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 33 Tests gruen
- echter Phase-3A-Lauf `real-phase3a-storyboard-1` erfolgreich verifiziert

## 2026-04-11 Tagesabschluss
- kanonisches Projektgedaechtnis unter `/workspace/codex` auf den echten Phase-3A-Endstand geschaerft
- `HANDOFF.md` fuer die naechste Session angelegt
- `.gitignore` vorsichtig um Laufzeit-/Artefaktordner, lokale Logs, Checkpoints, Legacy-Ordner und Egg-Info erweitert
- aktueller Tagesendstand erneut per `python -m unittest discover -s /workspace/tests -v` verifiziert -> 33 Tests gruen

## 2026-04-16 Phase-3B Keyframe Video Path
- vorhandenen Pod- und Backend-Stack gezielt auf ehrlichen keyframe-gestuetzten Video-Pfad geprueft
- bestaetigt: der bestehende FastAPI-/LTX2-Wrapper unterstuetzt im stabilen `ti2vid`-Pfad produktives Image-Conditioning via `--image`
- bewusst kein neuer Backend-Zweig und keine Fake-Keyframe-Interpolation gebaut
- `JobInput` um `video_mode` erweitert; `ScenePlan`, `TakePlan` und `TakeResultRecord` dokumentieren jetzt `video_mode`, `render_mode`, `fallback_strategy` und Laufzeit-`fallback_reason`
- Planner entscheidet jetzt pro Job oder optional pro Szene via `metadata.scene_video_modes`, ob `text_only`, `storyboard_reference` oder `keyframe_conditioned` geplant wird
- LTX2-Adapter injiziert den selektierten Storyboard-Keyframe jetzt produktiv als First-Frame-Image-Conditioning in den bestehenden `ti2vid`-Pfad
- Agent und Persistenz schreiben jetzt `selected_keyframe_usage`, `render_mode_counts` und `fallback_reasons` in `takes.json`, `state.json` und `result.json`
- neue Tests fuer keyframe-aware Planung, Fallback, Rendermodus-Persistenz und Multi-Scene-/Multi-Take-Kompatibilitaet hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 36 Tests gruen
- echter Phase-3B-Lauf `real-phase3b-keyframe-1` erfolgreich verifiziert:
  - Z-Image erzeugte zwei reale Keyframe-Kandidaten
  - LTX2 wurde real mit `--image` aus dem selektierten Keyframe gestartet
  - `render_mode=keyframe_conditioned` und `selected_keyframe_usage.applied=true` wurden im Job-Workspace persistiert

## 2026-04-16 Phase-4A Minimal Worker Bridge
- bestehenden Pod- und FastAPI-Stack auf kleinste saubere Aussenintegration geprueft
- bewusst eine duenne lokale FastAPI-Bridge statt neuer CLI-Familie oder grosser API-Plattform gewaehlt
- neuen Router `app/agent_core_api.py` eingefuehrt
- neuer synchroner Endpunkt `POST /agent-core/run` nimmt strukturierte Jobdaten entgegen und startet den bestehenden `VideoAgent`
- neuer Status-/Result-Endpunkt `GET /agent-core/jobs/{job_id}` liest den persistierten Jobzustand sauber zurueck
- `app.main` um den neuen Router und den statischen Mount `/agent-runs` erweitert, damit `state.json`, `result.json` und `final.mp4` auch direkt referenzierbar sind
- Beispielrequest `examples/agent_core_bridge_request.json` hinzugefuegt
- neue API-Tests fuer Job-Entry, Erfolg, Fehler und Validierungsfehler hinzugefuegt
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 40 Tests gruen
- echter lokaler HTTP-Lauf erfolgreich verifiziert:
  - `uvicorn app.main:app --port 8010`
  - `POST /agent-core/run` mit `bridge-demo-job`
  - `GET /agent-core/jobs/bridge-demo-job`
  - Rueckgabevertrag enthielt `result.output_final_path`, `refs.result_json_url` und `refs.final_mp4_url`

## 2026-04-16 Phase-4A Live Bridge Activation
- Ursache des Live-Problems auf Port `8000` verifiziert: der laufende `uvicorn app.main:app` war vor den Bridge-Aenderungen gestartet und kann im Pod nicht automatisch reloaden
- Codezustand gegen Live-Prozess gegengeprueft:
  - `app/agent_core_api.py` enthielt den Router korrekt
  - `app/main.py` band den Router korrekt ein
  - ein frischer Python-Import sah `/agent-core/run`, der Live-Server auf `8000` aber noch nicht
- produktiven FastAPI-Prozess auf Port `8000` manuell mit aktuellem Code neu gestartet
- Live-Router danach real verifiziert:
  - `GET /agent-core/run` auf `127.0.0.1:8000` liefert korrekt `405`, also kein `404` mehr
  - echter synchroner Run `POST http://127.0.0.1:8000/agent-core/run` erfolgreich mit `phase4a-live-verify-1776342448`
  - echter Statusabruf `GET http://127.0.0.1:8000/agent-core/jobs/phase4a-live-verify-1776342448` erfolgreich
  - Proxy-Pruefung erfolgreich:
    - `GET https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/jobs/phase4a-live-verify-1776342448`
    - `POST https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/run` liefert fuer `{}` korrekt `422`
- reales Finalartefakt bestaetigt: `/workspace/agent_runs/phase4a-live-verify-1776342448/final.mp4` mit `768x448`, `24 fps`, `4.042s`

## 2026-04-16 Phase-4B Async Polling Bridge
- bestehende Phase-4A-Bridge gezielt in Richtung minimalem Async-/Polling-Vertrag erweitert, ohne den `agent_core` umzubauen
- neuer produktiver Submit-Endpunkt `POST /agent-core/jobs` eingefuehrt
- `POST /agent-core/run` bewusst als synchroner Dev-/Test-Pfad beibehalten
- kleiner in-process Background-Runner im FastAPI-Router eingefuehrt; bewusst keine Queue-, Auth- oder Multi-User-Schicht gebaut
- Statusvertrag von `GET /agent-core/jobs/{job_id}` auf `accepted`, `queued`, `running`, `done` und `failed` geschaerft
- Statusantworten enthalten jetzt zusaetzlich `current_phase` und `poll_url`
- Polling-Pfad gegen kurzzeitig unvollstaendige JSON-Writes auf `state.json`/`result.json` gehaertet
- API-Tests auf Async-Annahme, laufenden Status, Erfolg und Fehlerpfad erweitert
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 41 Tests gruen
- produktiven FastAPI-Prozess auf Port `8000` nach den Phase-4B-Aenderungen manuell neu geladen
- echter produktiver Async-Lauf erfolgreich verifiziert:
  - `POST http://127.0.0.1:8000/agent-core/jobs` mit `phase4b-live-verify-1776343554`
  - Polling ueber `GET http://127.0.0.1:8000/agent-core/jobs/phase4b-live-verify-1776343554`
  - Endstatus `done`, `current_phase=done`, `result.final_phase=assembled`
  - Proxy-Statusabruf ueber `https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/jobs/phase4b-live-verify-1776343554` erfolgreich
  - reales Finalartefakt bestaetigt: `/workspace/agent_runs/phase4b-live-verify-1776343554/final.mp4`

## 2026-04-16 Phase-4C n8n-Friendly Polling Hardening
- bestehenden Async-/Polling-Vertrag gezielt fuer n8n gehaertet, ohne den Core oder die Produktionslogik umzubauen
- `GET /agent-core/jobs/{job_id}` liefert jetzt zusaetzlich:
  - `status_summary`
  - `is_terminal`
  - `should_poll`
  - `retry_after_sec`
  - `artifacts_ready`
  - `final_mp4_ready`
  - `result_json_ready`
  - `public_refs`
- `public_refs` fuehrt nur die extern nutzbaren URLs fuer `state.json`, `result.json` und `final.mp4`
- Fehljobs exponieren keinen irrefuehrenden `final_mp4`-Public-Link mehr, auch wenn lokal Zwischenartefakte liegen
- `failed` kann jetzt im Polling-Vertrag frueh sichtbar sein, bleibt aber fuer n8n erst terminal, wenn der Failure-Vertrag wirklich bereit ist
- API-Tests um Assertions fuer Terminal-Flags, Polling-Hinweise und Artefakt-Readiness geschaerft
- Tests erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 41 Tests gruen
- produktiven FastAPI-Prozess auf Port `8000` nach den Phase-4C-Aenderungen manuell neu geladen
- realer Live-Response-Check erfolgreich:
  - `POST http://127.0.0.1:8000/agent-core/jobs` mit `phase4c-live-verify-1776348348`
  - verifizierte Submit-Felder: `accepted`, `is_terminal=false`, `should_poll=true`, `retry_after_sec=2`
  - verifizierter Mid-Poll: `running`, `result_json_ready=false`, `final_mp4_ready=false`
  - verifizierter Final-Poll: `done`, `is_terminal=true`, `should_poll=false`, `artifacts_ready=true`
  - Proxy-Response ueber `https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/jobs/phase4c-live-verify-1776348348` erfolgreich
