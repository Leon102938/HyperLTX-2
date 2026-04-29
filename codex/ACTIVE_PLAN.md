# ACTIVE_PLAN.md

## Aktiver Gesamtplan
Phase 1: Agent-Core-Kern
Phase 2A: Scene-/Shot-Planung
Phase 2B: Mehrfach-Takes und Auswahl pro Szene
Phase 2C: technischer Quality-Guard, validierte Auswahl und leichte Retries pro Szene
Phase 2D: Shot-/Prompt-Variation-Engine pro Szene
Phase 2E: leichte inhaltliche Varianten-/Take-Auswahl ueber dem technischen Vertrag
Phase 3A: optionale Storyboard-/Keyframe-Pipeline
Phase 3B: produktive Keyframe-Nutzung im bestehenden Video-Pfad
Phase 4A: minimale Worker-/n8n-Bridge fuer den bestehenden Core
Phase 4B: minimale asynchrone Job-Bridge mit Polling-Vertrag
Phase 4C: kleine n8n-friendly Polling-Haertung
Phase 5A: Director-/Brain-Schicht fuer bessere Regie, Planung und Prompts
Phase 5B: echter lokaler Director-LLM-Pfad
Phase 6: Ausbau, Optimierung, Spezialisierung

## Phase 5A Ziel
Eine kleine Director-/Brain-Schicht vor dem bestehenden Planner bauen, die Jobs in staerkere kreative Briefs, konsistente Style-Locks, klarere Szenenintents und bessere Prompt-Bausteine uebersetzt, ohne den vorhandenen Core oder den produktiven Video-Pfad gross umzubauen.

## Phase 5B Ziel
Den bereits vorbereiteten Director-Layer produktiv an ein echtes lokales Director-Modell anbinden, bevorzugt Qwen3.6-35B-A3B in praktikabler quantisierter Form als GGUF `Q4_K_M`, ohne Fake-Integration, ohne neuen Mega-Stack und mit sauberem Fallback auf den bisherigen regelbasierten Flow.

## Aktueller Operativer Fokus
- Aktueller Output-Quality-Fokus Phase A ist umgesetzt: Scene World Contract + PromptBuilder v2 haerten Szene- und Variation-Prompts gegen Text-/Screen-/Papier-Artefakte, ohne Runtime-/Backend-Aenderung.
- Phase B1 ist umgesetzt: Storyboard-/Keyframe-Kandidaten erhalten jetzt scene-specific, contract-aware `effective_prompt`-Metadaten; Z-Image nutzt diese bevorzugt statt nur globaler Plan-Prompts.
- Phase B2 ist umgesetzt: Storyboard-Keyframe-Kandidaten erhalten jetzt einen leichten `visual_risk_review` mit Status `passed`, `needs_review` oder `rejected`; Auswahl bevorzugt `passed` vor `needs_review` vor `rejected`.
- Phase B1/B2 sind per Unit-Tests und Dry-Run-Artefaktplaenen verifiziert; noch kein neuer langer GPU-Render und keine finale visuelle Qualitaetsbehauptung.
- Naechster Output-Quality-Schritt: Phase C Take Visual Review / Postability Score.
- Spaetere geplante Schritte: Phase D Final Quality Verdict, Phase E CLI Produktions-Cockpit.
- Restore-/Startup-/Environment-Check fuer den lokalen Director-Pfad ist jetzt auch ueber einen frischen `bash /workspace/init.sh`-Lauf real verifiziert; `init.sh` brachte `llama-server` dabei ohne manuellen Director-Start selbst hoch.
- Der vorhandene `llama.cpp`-Runtime-/Build-Stand unter `/workspace/tools/llama.cpp/build/bin` ist jetzt nicht nur ohne Rebuild verifiziert, sondern wird im aktuellen Workspace vor einem Rebuild zuerst ueber Execute-Bits und Linux-Symlink-Aliase repariert.
- Naechster sinnvolle Ausbaupunkt bleibt unveraendert: weitere echte Multi-Scene-/Storyboard-Validierung des bestehenden Qwen-Director-Pfads, kein neuer Feature-Sprung.
- Vor dem naechsten Abschluss oder Backup muss die Dateiliste explizit auf Vollstaendigkeit gegen den realen Director-/Startup-Pfad geprueft werden, insbesondere `tools/llama.cpp`, `config/director_llm.env`, neue Director-Skripte sowie Fixes in `start.sh`, `init.sh` und `app/main.py`.

## Phase 5A Arbeitsplan
1. Kleinste saubere Integrationsstelle im `ProductionPlanner` nutzen, statt den bestehenden `agent_core` gross zu refactoren.
2. Neue Module fuer `director`, `llm_adapter`, `prompt_builder` und `style_memory` einfuehren.
3. `director_output`, `style_lock`, `scene_intent`, `creative_intent` und `prompt_build_metadata` in die bestehenden Artefakte einspeisen.
4. Einen optionalen lokalen OpenAI-kompatiblen Director-Adapter bauen, aber ohne Fake-Behauptungen oder Modell-Download-Orgie.
5. Saubere Fallbacks auf den bisherigen regelbasierten Flow behalten und den aktiven Modus explizit persistieren.
6. Tests fuer Struktur, Fallback, Prompt-Bau, Persistenz und Flow-Kompatibilitaet ausfuehren.
7. Einen ehrlichen Real-Check mit aktivem Director-Layer im verfuegbaren Modus dokumentieren.

## Empfohlener Minimal-Vertical-Slice
- Eingabe: einfacher Job mit `prompt`, optional `tts`, optional `audio`, optional `video`
- Ablauf:
  1. Job validieren
  2. Plan erzeugen
  3. Step-State persistieren
  4. Qwen-TTS optional ausfuehren
  5. LTX-2 Step ausfuehren
  6. finales MP4 assembliert speichern

## Phase-5A-Ist-Stand
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
  - Auswahlregel `quality_guarded_best_valid_take`
  - `takes.json` als Take-Artefakt
  - technischer Quality-Guard pro Take
  - begrenzte Retry-Regeln pro Szene
  - regelbasierte Shot-/Prompt-Variations-Engine pro Szene
  - Variantenvertrag fuer Szene, Take und Persistenz
  - regelbasierte kreative Auswahlheuristik ueber technisch gleichwertigen validen Kandidaten
  - neue Auswahlmetadaten `technical_score`, `creative_score`, `selection_reason`, `selected_by_rule`
  - optionaler Storyboard-Step auf Basis von Z-Image
  - `storyboard_plan.json` als neues Storyboard-Artefakt
  - Storyboard-Konfiguration, Keyframe-Kandidaten und selektierte Keyframes pro Szene
  - `video_mode` auf Job-Ebene plus `render_mode`-/Fallback-Vertrag auf Szene- und Take-Ebene
  - produktive Keyframe-Nutzung im bestehenden LTX2-`ti2vid`-Pfad via First-Frame-Image-Conditioning
  - Persistenz fuer `selected_keyframe_usage`, `render_mode_counts` und `fallback_reasons`
  - neuer duennner FastAPI-Router `app/agent_core_api.py`
  - synchroner Run-Endpunkt `POST /agent-core/run`
  - asynchroner Submit-Endpunkt `POST /agent-core/jobs`
  - Status-/Result-Endpunkt `GET /agent-core/jobs/{job_id}`
  - statische Referenzierung von `agent_runs` ueber `/agent-runs`
  - kanonische Bridge-Beispielaufrufe jetzt direkt in `codex/COMMAND_PROMPTS.md` als Inline-JSON
  - kleiner in-process Background-Runner fuer Async-Submits
  - n8n-freundliche Polling-Felder `status_summary`, `is_terminal`, `should_poll`, `retry_after_sec`
  - explizite Artefakt-Readiness `artifacts_ready`, `final_mp4_ready`, `result_json_ready`
  - neues `public_refs`-Objekt nur fuer extern nutzbare URLs
  - neue Director-Module `agent_core/director.py`, `agent_core/llm_adapter.py`, `agent_core/prompt_builder.py`, `agent_core/style_memory.py`
  - `director_output.json` als neues Regie-Artefakt
  - `director_output`, `scene_intent`, `creative_intent` und `prompt_build_metadata` im bestehenden Plan-/Scene-/Take-/Result-Vertrag
  - optionaler lokaler OpenAI-kompatibler Director-Adapter mit ehrlichem Fallback auf `rule_based_fallback`
  - `app/agent_core_api.py` im Workspace wiederhergestellt und in `app.main` eingebunden
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
  - jeder erfolgreiche Take traegt jetzt `review_status` plus `validation`-Block
  - nur validierte selektierte Takes werden assembliert
  - Retry-Historie wird pro Szene in `takes.json` und `state.json` dokumentiert
  - echter Phase-2C-Lauf erfolgreich mit `real-phase2c-quality-guard-1`
  - pro Szene koennen mehrere kreative Varianten geplant werden
  - jede Variation fuehrt zu eigenen renderbaren Takes
  - `scene_plan.json` und `takes.json` dokumentieren jetzt Varianten und die ausgewaehlte Variation
  - echter Phase-2D-Lauf erfolgreich mit `real-phase2d-variation-1`
  - kreativ gleichwertige technische Kandidaten koennen jetzt regelbasiert aufgeloest werden
  - benachbarte Szenen vermeiden unnoetige Shot-Typ-Wiederholung
  - Auswahlmetadaten werden in `takes.json`, `state.json` und Result-Metadaten persistiert
  - echter Phase-2E-Lauf erfolgreich mit `real-phase2e-creative-selection-1`
  - echter Phase-3A-Lauf erfolgreich mit `real-phase3a-storyboard-1`
  - Storyboard-Kandidaten werden technisch validiert und selektiert
  - selektierte Keyframes werden in `storyboard_plan.json`, `state.json`, `result.json` und `takes.json` dokumentiert
  - echter Phase-3B-Lauf erfolgreich mit `real-phase3b-keyframe-1`
  - der reale LTX2-Job wurde dabei mit dem selektierten Storyboard-Keyframe als `--image` im bestehenden `ti2vid`-Pfad gestartet
  - `takes.json`, `state.json` und `result.json` dokumentieren jetzt produktiv, ob `keyframe_conditioned`, `storyboard_reference` oder `text_only` aktiv war
  - neuer Bridge-Test `tests/test_agent_core_api.py` ist gruen
  - echter lokaler HTTP-Lauf ueber `POST /agent-core/run` erfolgreich mit `bridge-demo-job`
  - echter lokaler Statusabruf ueber `GET /agent-core/jobs/bridge-demo-job` erfolgreich
  - `POST /agent-core/jobs` liefert jetzt sofort `202 Accepted`, `job_id` und `poll_url`
  - `GET /agent-core/jobs/{job_id}` liefert jetzt polling-faehig `accepted`, `queued`, `running`, `done` oder `failed`
  - der Statuspfad ist gegen kurzzeitig unvollstaendige JSON-Writes gehaertet
  - echter produktiver Async-Lauf erfolgreich mit `phase4b-live-verify-1776343554`
  - echter Proxy-Statusabruf fuer den Async-Pfad erfolgreich
  - Polling-Antworten liefern jetzt stabil `is_terminal` und `should_poll` fuer n8n
  - `done`- und `failed`-Antworten exponieren Artefakt-Readiness jetzt explizit statt nur implizit ueber `null`
  - ein verifizierter Fehljob zeigt kein irrefuehrendes `final.mp4` mehr als public ready an
  - echter Live-Response-Check erfolgreich mit `phase4c-live-verify-1776348348`
  - Director-Ausgabe-Struktur, Fallback, Prompt-Building, Persistenz und Flow-Kompatibilitaet sind durch `tests/test_director_layer.py` plus die bestehenden Planner-/Smoke-/Storyboard-Tests abgesichert
  - kompletter Testlauf aktuell erfolgreich: `python -m unittest discover -s /workspace/tests -v` -> 49 Tests gruen
  - echter Phase-5A-Live-Fallback erfolgreich mit `phase5a-live-fallback-1776420785`
  - der reale Director-Modus dieses Live-Laufs war `rule_based_fallback`; ein lokaler Director-LLM-Dienst war im Pod nicht produktiv verfuegbar
  - Tagesabschluss-Doku und Handoff werden in `/workspace/codex` gepflegt

## Phase-5B-Ist-Stand
- gebaut:
  - echter lokaler Qwen3.6-35B-A3B-Serving-Pfad ueber `llama.cpp` + GGUF
  - Build-Pfad fuer `llama-server` unter `/workspace/tools/llama.cpp/build/bin/llama-server`
  - Modellpfad-Konvention unter `/workspace/models/director/qwen3.6-35b-a3b/gguf/`
  - `scripts/download_director_model.py`, `scripts/serve_director_llm.sh`, `scripts/check_director_llm.py`
  - `config/director_llm.env.example`
  - reale lokale Default-Konfiguration `config/director_llm.env`
  - neues lokales Director-Profil `qwen36_llama_cpp_local` im `llm_adapter`
  - explizite Director-LLM-Statusfelder `llm_active`, `llm_provider`, `llm_model`, `llm_endpoint`
  - echte Normalisierung eines kleineren `scene_map`-LLM-Outputs in den bestehenden `DirectorOutput`-Vertrag
  - idempotente Director-Modell-Vorbereitung in `init.sh` mit optionalem Auto-Start des lokalen Serve-Diensts
  - kleine Serve-Haertung fuer `llama-server`: konfigurierbare Health-Checks, Readiness-Retries, PID-Bereinigung und fruehes Abbrechen bei Startfehlern
  - kleiner Rebuild-Guard in `scripts/ensure_llama_cpp.sh`, der bei Bedarf auch `ninja` installiert
- verifiziert:
  - lokaler OpenAI-kompatibler Endpoint antwortet real auf `http://127.0.0.1:8011/v1/chat/completions`
  - Qwen3.6 laeuft real als `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - Director-Planung kann jetzt real `llm_augmented` statt `rule_based_fallback` liefern
  - ein Low-Memory-Profil `-ngl 8 -c 2048 --reasoning off --no-warmup` erlaubt echten erfolgreichen Agent-Run mit aktivem Director-LLM
  - echter erfolgreicher Live-Run: `phase5b-qwen-live-1776506522`
  - `result.json` exponiert dort real `director_llm_active`, `director_llm_provider`, `director_llm_model` und `director_llm_endpoint`
  - Restore-/Startup-Check nach Repo-Update und Pod-Neustart ist mit realem Async-FastAPI-Lauf erneut verifiziert: `restore-startup-check-20260418` via `POST /agent-core/jobs`, Director-Modus `llm_augmented`, finales MP4 `/workspace/agent_runs/restore-startup-check-20260418/final.mp4`
  - Director-Defaultpfad aus `config/director_llm.env` ist ebenfalls real verifiziert: `director-stability-check-20260418` via `POST /agent-core/jobs`, Director-Modus `llm_augmented`, finales MP4 `/workspace/agent_runs/director-stability-check-20260418/final.mp4`

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
- grosse Multi-User-API fuer den Core
- Queue-System oder asynchrones Job-Management fuer die Bridge
- Auth-System
- n8n-spezifische Speziallogik
- GUI
- Multi-Agent-Swarm
- DB, Queue-Cluster, Event-Bus
- tiefe Umbauten in `LTX-2`, `ACE-Step-1.5` oder `Qwen3-TTS`
- vollwertige Musik-, Hook- oder Quality-Subsysteme
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

## Phase-5A-Abschlusskriterium
- Der bestehende Planner kann jetzt mit aktiver Director-Schicht `creative_brief`, `style_lock`, `scene_intent`, staerkere Variationen und kompaktere Prompts erzeugen, ohne den bisherigen Scene-/Take-/Storyboard-/Assembler-Vertrag zu brechen.
- Wenn kein lokaler Director-LLM-Dienst produktiv verfuegbar ist, faellt der Core ehrlich und dokumentiert auf `rule_based_fallback` zurueck.

## Phase-5B-Abschlusskriterium
- Der Director-Layer kann jetzt gegen einen echten lokalen Qwen3.6-35B-A3B-Dienst produktiv `llm_augmented` fahren.
- Wenn der Dienst nicht erreichbar ist, bleibt der bestehende `rule_based_fallback` sauber aktiv.
- Die aktive Director-LLM-Nutzung, das Modell und eventuelle Fallback-Gruende werden persistiert.

## Tagesabschluss
- Commit-wuerdig sind aktuell die Quellordner `agent_core/`, `app/`, `tests/`, `examples/`, die geschraefte `.gitignore` und der kanonische Projekt-Memory unter `/workspace/codex`.
- Laufzeit- und Artefaktordner wie `agent_runs/`, `exports/`, `jobs/`, `status/`, `venvs/` und Checkpoints bleiben bewusst ausserhalb eines sauberen Code-Commits.
