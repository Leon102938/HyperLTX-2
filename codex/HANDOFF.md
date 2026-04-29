# HANDOFF.md

## Aktueller Gesamtstand
- Git-Root ist `/workspace`.
- Kanonischer Projekt-Memory-Pfad ist `/workspace/codex`.
- Kanonische Capability-Uebersicht liegt jetzt in `/workspace/codex/CAPABILITY_MAP.md`.
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
- Phase 3B ist abgeschlossen:
  - produktiver First-Frame-Keyframe-Pfad im bestehenden LTX2-`ti2vid`-Stack
  - `video_mode`, `render_mode`, `fallback_strategy` und `selected_keyframe_usage` werden pro Job/Szene/Take persistiert
  - ehrlicher Fallback auf `storyboard_reference` oder `text_only`, wenn der selektierte Keyframe nicht nutzbar ist
- Phase 4A ist abgeschlossen:
  - minimale produktive Worker-/n8n-Bridge ueber die bestehende FastAPI
  - `POST /agent-core/run` startet den bestehenden `VideoAgent` synchron
  - `GET /agent-core/jobs/{job_id}` liefert den persistierten Status-/Result-Vertrag
  - `/agent-runs` ist als Referenzpfad fuer `state.json`, `result.json` und `final.mp4` gemountet
  - der Live-Server auf Port `8000` wurde nach den Bridge-Dateiaenderungen manuell neu geladen und der Endpunkt ist dort jetzt real verifiziert
- Phase 4B ist abgeschlossen:
  - produktiver Async-Submit-Pfad `POST /agent-core/jobs` ist gebaut
  - `GET /agent-core/jobs/{job_id}` liefert aktuell polling-faehig `accepted`, `running`, `done` oder `failed`
  - `POST /agent-core/run` bleibt als synchroner Dev-/Test-Pfad erhalten
  - der Live-Server auf Port `8000` wurde nach den Phase-4B-Aenderungen erneut manuell neu geladen und der Async-Pfad ist dort jetzt real verifiziert
- Phase 4C ist abgeschlossen:
  - die Polling-Antworten sind jetzt n8n-freundlicher gehaertet
  - `status_summary`, `is_terminal`, `should_poll`, `retry_after_sec`, `artifacts_ready`, `final_mp4_ready`, `result_json_ready` und `public_refs` sind jetzt im Vertrag
  - Fehljobs exponieren keinen irrefuehrenden finalen Public-Link mehr
  - der Live-Server auf Port `8000` wurde nach den Phase-4C-Aenderungen erneut manuell neu geladen und die neuen Felder sind dort jetzt real verifiziert
- Phase 5A ist abgeschlossen:
  - eine kleine Director-/Brain-Schicht sitzt jetzt vor dem bestehenden Planner
  - `director_output`, `style_lock`, `scene_intent`, `creative_intent` und `prompt_build_metadata` werden in Plan-/Scene-/Take-/Result-Artefakte eingespeist
  - staerkere Opening-Shots, klarere visuelle Sprache und konsistentere Varianten werden ueber `prompt_builder.py` erzeugt
  - ein optionaler lokaler OpenAI-kompatibler Director-Adapter ist gebaut; ohne produktiven Dienst faellt der Core ehrlich auf `rule_based_fallback` zurueck
  - `app/agent_core_api.py` ist im Workspace wieder real vorhanden und die Phase-4-Bridge-Tests sind wieder gruen
- Phase 5B ist jetzt in einem echten produktiven Minimalpfad umgesetzt:
  - lokaler Director-Serve ueber `llama.cpp` + GGUF statt Placeholder
  - reales Zielmodell `Qwen3.6-35B-A3B` als `Q4_K_M`
  - neues Profil `qwen36_llama_cpp_local`
  - Serve-/Download-/Smoke-Skripte vorhanden
  - `init.sh` bereitet das Modell jetzt idempotent vor und kann den lokalen Serve optional automatisch starten
  - vor dem Director-Setup werden alte Ready-Flags jetzt geloescht, damit der Status nicht stale wird
  - Director-LLM-Nutzung, Modell und Endpoint werden explizit in `director_output.json` und `result.json` persistiert
  - verifizierter erfolgreicher Live-Run mit aktivem Director-LLM: `phase5b-qwen-live-1776506522`
  - wichtige Einordnung: die dort sichtbaren `320x256` stammen aus einem explizit klein gesetzten Verifikationsjob; der Default-Renderpfad blieb unveraendert bei `resolution="standard"`
- Restore-/Startup-Check nach Repo-Update und Pod-Neustart ist jetzt ebenfalls real verifiziert:
  - der konkrete FastAPI-Crash durch fehlendes `/workspace/agent_runs` ist minimal gefixt
  - `app.main`, `start.sh` und `init.sh` erzeugen die Basis-Laufzeitordner jetzt vor Zugriff bzw. idempotent
  - `/workspace/tools/llama.cpp/build/bin/llama-server` fehlte nach dem Restore zunaechst wieder, wurde aber ueber `scripts/serve_director_llm.sh` real neu gebaut
  - `config/director_llm.env` ist jetzt real vorhanden und wird von `start.sh`, `init.sh`, `app.main` und `scripts/check_director_llm.py` konsistent geladen
  - `scripts/serve_director_llm.sh` ist jetzt mit kleinen Health-/Retry-/PID-Guards gehaertet; `scripts/ensure_llama_cpp.sh` sichert bei Restore-Rebuilds jetzt auch `ninja`
  - ein zusaetzlicher echter Restore-/Startup-Fall wurde am 2026-04-20 belegt: `cli-test-basic-001` fiel auf `rule_based_fallback`, weil `init.sh` den Director-Autostart uebersprang, als `serve_director_llm.sh` zu diesem Zeitpunkt noch nicht executable war
  - minimaler Fix dafuer in `init.sh`: vor dem Director-Autostart wird `serve_director_llm.sh` jetzt explizit `chmod +x` gesetzt und per `bash` gestartet
  - der direkte Healthcheck danach war wieder gruen: `/v1/models` und `python3 /workspace/scripts/check_director_llm.py`
  - neuer echter enger Recheck ueber den produktiven CLI/API-Pfad: `cli-test-basic-001-reverify` lief danach wieder mit `director_mode=llm_augmented`
  - neuer echter Restore-Live-Run mit aktivem Director-LLM: `restore-startup-check-20260418`
  - neuer echter Defaultpfad-Live-Run mit aktivem Director-LLM: `director-stability-check-20260418`
  - der kleine Restore-Folgecheck `restore-health-check-20260419` lief real mit `llm_augmented`, scheiterte danach aber in LTX2 an einem CUDA-OOM
  - `DIRECTOR_LLM_N_GPU_LAYERS` wurde danach minimal von `8` auf `0` gesenkt, um den Director-GPU-Footprint fuer den produktiven Kombipfad weiter zu entlasten
  - neuer echter Minimal-Live-Run nach diesem GPU-Fix: `director-gpu-fix-check-20260419`
  - neuer echter kleiner Voice-Live-Run auf dem produktiven API-Pfad: `voice-stability-check-20260419`
  - dieser Voice-Run lief real mit `director_mode=llm_augmented`, `director_llm_active=true`, Qwen-TTS, LTX2 und erzeugte erfolgreich `final.mp4`
  - fuer schnelle lokale Manual-Checks gibt es jetzt zusaetzlich `scripts/agent_core_cli.py` als kleinen Submit-/Polling-Wrapper ueber denselben produktiven API-Vertrag
  - `tests/test_director_layer.py` isoliert `DIRECTOR_LLM_*` jetzt bewusst im Test-Setup, damit lokale Pod-Defaults die Fallback-Tests nicht kippen
  - neuer enger frischer Startup-Recheck am 2026-04-21:
    - kompletter Pod-Neustart war in der Session nicht praktikabel; der engste reale Startpfad wurde deshalb direkt ueber `bash /workspace/init.sh` erneut ausgefuehrt
    - vor dem Recheck war `127.0.0.1:8011` real down, waehrend FastAPI auf `8000` lief
    - der frische `init.sh`-Lauf hat den Director danach ohne manuelles `serve_director_llm.sh` selbst gestartet
    - direkt danach waren `ps`, `/v1/models` und `python3 /workspace/scripts/check_director_llm.py` gruen
    - neuer echter enger Produktivcheck danach: `startup-recheck-20260421`
    - dieser Lauf lief real mit `director_mode=llm_augmented`, `director_llm_active=true` und erzeugte erfolgreich `/workspace/agent_runs/startup-recheck-20260421/final.mp4`
    - wichtiger Abschluss fuer diesen Punkt: der minimale `init.sh`-Autostart-Fix ist damit nicht mehr nur wahrscheinlich, sondern ueber einen frischen `init.sh`-Startpfad real verifiziert
- Output-Quality-Fokus Phase A/B1/B2 ist umgesetzt:
  - Phase A: Scene World Contract + PromptBuilder v2; Scene-/Variation-Prompts enthalten harte Text-/Screen-/Paper-Verbote
  - Phase B1: Storyboard-/Keyframe-Prompts sind scene-specific und contract-aware; Z-Image nutzt `effective_prompt`
  - Phase B2: Storyboard-Keyframe-Kandidaten erhalten `visual_risk_review` mit `passed`, `needs_review` oder `rejected`; Auswahl bevorzugt `passed` vor `needs_review` vor `rejected`
  - Dry-Run-Artefakte liegen unter `/workspace/agent_runs/phase-b1-dry-*` und `/workspace/agent_runs/phase-b2-dry-*`
  - keine finale Bildqualitaet behauptet; naechster Schritt ist Phase C Take Visual Review / Postability Score

## Was real verifiziert wurde
- Tests:
  - `python -m unittest discover -s /workspace/tests -v` -> 49 Tests gruen
  - `python3 -m unittest tests.test_director_layer -v` -> 7 Tests gruen nach Env-Isolation der Director-Tests
- Kleiner Real-Check:
  - `python3 /workspace/scripts/download_director_model.py` meldet bei vorhandenem Modell sauber nur `present: ...` und startet keinen unnötigen Neu-Download
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
  - `real-phase3b-keyframe-1`
  - `bridge-demo-job` via `POST /agent-core/run`
  - `phase4a-live-verify-1776342448` via `POST http://127.0.0.1:8000/agent-core/run`
  - `phase4b-live-verify-1776343554` via `POST http://127.0.0.1:8000/agent-core/jobs`
  - `phase4c-live-verify-1776348348` via `POST http://127.0.0.1:8000/agent-core/jobs`
  - `phase5a-live-fallback-1776420785` via direktem `VideoAgent()`-Run mit aktivem Director-Layer im `rule_based_fallback`-Modus
  - `phase5b-qwen-live-1776506522` via direktem `VideoAgent()`-Run mit aktivem Qwen3.6-Director; `result.json` und `director_output.json` dokumentieren dort den aktiven Director-LLM-Pfad
  - `restore-startup-check-20260418` via `POST http://127.0.0.1:8000/agent-core/jobs` mit erneut real verifiziertem `llm_augmented`-Director-Pfad
  - `director-stability-check-20260418` via `POST http://127.0.0.1:8000/agent-core/jobs` mit Director-Defaults aus `config/director_llm.env` und erneut real verifiziertem `llm_augmented`-Pfad
  - `director-gpu-fix-check-20260419` via `POST http://127.0.0.1:8000/agent-core/jobs` mit erneut real verifiziertem `llm_augmented`-Director-Pfad und erfolgreichem kleinem LTX2-Render nach dem GPU-Fix
  - `voice-stability-check-20260419` via `POST http://127.0.0.1:8000/agent-core/jobs` mit aktivem Voice-Pfad, `llm_augmented`, erfolgreichem Qwen-TTS-Render, erfolgreichem LTX2-Video und finalem muxed `final.mp4`
  - `startup-recheck-20260421` via frischem `bash /workspace/init.sh` plus `POST http://127.0.0.1:8000/agent-core/jobs` mit erneut real verifiziertem `llm_augmented`-Director-Pfad und erfolgreichem `final.mp4`
- Reale lokale Backends verifiziert:
  - Qwen TTS ueber vorhandene FastAPI-Endpunkte
  - LTX2 `ti2vid` ueber vorhandene FastAPI-Endpunkte
  - Z-Image ueber vorhandene FastAPI-Endpunkte
  - lokaler Director-LLM-Serve ueber `llama.cpp` auf `127.0.0.1:8011`
- Reale Quality-Verifikation am 2026-04-21:
  - die alten Artefaktordner `demo-social-morning-003` bis `005` fehlten im aktuellen Workspace; die Baseline musste deshalb ehrlich ueber die kanonische Doku plus neue Realruns gezogen werden
  - `demo-social-morning-006` lief noch ueber einen stale Live-Server und zeigte in `plan.json` weiter `social_tip_visual_guard_version=v1`; der produktive Uvicorn auf `8000` musste fuer echte Live-Verifikation manuell neu gestartet werden
  - `demo-social-morning-007` lief danach real mit `director_mode=llm_augmented`, `social_tip_visual_guard_version=v2`, `social_tip_visual_guard_family=morning_reset` und `final.mp4`
  - `demo-social-morning-007` zeigt real die neue robustere Szene 3 als Kuechenroutine statt generischem neutralem B-Roll; Overlay und Burn-in-Subtitles bleiben funktionsfaehig
  - `demo-social-focus-break-001` lief real mit der neuen `focus_break`-Szenenfolge, blieb aber visuell schlecht: Whiteboard-/Screen-/Textmuell dominierte trotz sauberer Szenenintents
  - minimaler produktiver Nachfix in `agent_core/planner.py`: Social-Tipp-Familien ueberschreiben jetzt auch `style_lock.visual_identity`, damit textnahe Director-Style-Locks nicht ungefiltert in den Prompt rutschen
  - `demo-social-focus-break-002` lief danach real mit dem neuen display-fernen `style_lock.visual_identity` und `final.mp4`; der sichtbare Screen-/Papier-/Glyph-Muell blieb aber weiterhin deutlich vorhanden
- Backup-Status dieses Schritts:
  - in diesem Schritt wurde bewusst kein neues Backup erzeugt
  - beim naechsten Abschluss oder Backup muss die Dateiliste explizit auf Vollstaendigkeit gegen den realen Director-/Startup-Pfad geprueft werden
  - dabei duerfen insbesondere diese Pfade nicht fehlen:
    - `/workspace/tools/llama.cpp`
    - `/workspace/config/director_llm.env`
    - `/workspace/scripts/download_director_model.py`
    - `/workspace/scripts/serve_director_llm.sh`
    - `/workspace/scripts/check_director_llm.py`
    - `/workspace/scripts/ensure_llama_cpp.sh`
    - `/workspace/start.sh`
    - `/workspace/init.sh`
    - `/workspace/app/main.py`

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
  - Phase 3B
  - Phase 4A
  - Phase 4B
  - Phase 4C
  - Phase 5A
  - Phase 5B
- Noch nicht gebaut:
  - groessere API-Plattform
  - eigentliche n8n-Orchestrierung/Queue-Integration
  - Musik-Pipeline
  - tieferer Character-/Voice-/World-Lock
  - zweiter produktiver Backend-Pfad im neuen Core

## Was als Naechstes sinnvoll ist
- Kleinster sinnvoller naechster Schritt:
  - nach dem jetzt real gruenen frischen `init.sh`-Startup-Recheck den bestehenden Qwen3.6-Director-Pfad ueber einen kleinen Multi-Scene- oder Storyboard-Live-Run weiter validieren
- Alternative:
  - Character-/Voice-/World-Lock kontrolliert ausbauen
- Nicht sinnvoll als naechster Schritt:
  - sofortige grosse Multi-User-API
  - sofortige Queue-/Auth-/Frontend-Schicht
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
- Phase 5B nutzt jetzt einen realen lokalen Qwen3.6-Director; die Director-Planung ist echt verifiziert, aber der lokale Serve bleibt spuerbar langsamer als der regelbasierte Fallback und sollte fuer weitere Ausbauschritte unter Last weiter beobachtet werden.
- Phase 3B nutzt selektierte Keyframes jetzt produktiv fuer First-Frame-Conditioning, aber noch nicht fuer Multi-Keyframe-Interpolation oder einen separaten Interpolations-/Retake-Vertrag.
- Phase 4A/4B/4C bleiben kleine Minimal-Bridges; Phase 4B ist zwar polling-faehig und 4C macht den Vertrag n8n-freundlicher, aber es gibt weiterhin keine durable Queue, keine Auth und keine eigentliche n8n-spezifische Steuerlogik.
- Multi-Segment-Concat kann noch kleine Timing-Deltas erzeugen.
- Der Worktree ist lokal deutlich verschmutzt durch Runtime- und Modellordner; saubere Commits muessen gezielt nur Code und Doku umfassen.
- `init.sh` ist bereits lokal modifiziert und nicht von dieser Session bereinigt worden.

## Wiederaufnahme-Prompt fuer die naechste Session
```text
Lies zuerst in /workspace/codex:
AGENTS.md, MISSION.md, USER_PREFERENCES.md, PROJECT_STATE.md, ACTIVE_PLAN.md, MEMORY.md, DECISIONS.md, CHANGELOG.md, TASK_BOARD.md, COMMAND_PROMPTS.md und HANDOFF.md.

Behandle nur /workspace/codex als kanonisches Projektgedaechtnis.
Nutze den bestehenden Phase-1-, 2A-, 2B-, 2C-, 2D-, 2E-, 3A-, 3B- und 4A-Stand unveraendert als Basis.
Keine n8n-Anbindung, keine externe API-Schicht, keine GUI und kein grosser Refactor, ausser der neue Auftrag verlangt das explizit.

Arbeite danach auf Basis verifizierter Fakten und aktualisiere die Memory-Dateien sauber.
```
