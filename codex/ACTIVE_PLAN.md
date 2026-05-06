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
Phase G1: Pipeline Definitions, Checkpoints und lokale Approval Gates
Phase G1.1: CLI Checkpoint Inspect und lokale Approval/Reject UX
Phase G2: Skill Layer, Pipeline Modes und Creative Roles
Phase 6: Ausbau, Optimierung, Spezialisierung

## Phase 5A Ziel
Eine kleine Director-/Brain-Schicht vor dem bestehenden Planner bauen, die Jobs in staerkere kreative Briefs, konsistente Style-Locks, klarere Szenenintents und bessere Prompt-Bausteine uebersetzt, ohne den vorhandenen Core oder den produktiven Video-Pfad gross umzubauen.

## Phase 5B Ziel
Den bereits vorbereiteten Director-Layer produktiv an ein echtes lokales Director-Modell anbinden, bevorzugt Qwen3.6-35B-A3B in praktikabler quantisierter Form als GGUF `Q4_K_M`, ohne Fake-Integration, ohne neuen Mega-Stack und mit sauberem Fallback auf den bisherigen regelbasierten Flow.

## Aktueller Operativer Fokus
- Creative OS CLI Dashboard V1 ist gebaut. Aktueller read-only Inspect-Befehl: `python3 /workspace/scripts/creative_os_status.py --job-id creative-os-jungle-001 --view overview`.
- Dashboard-Status fuer `creative-os-jungle-001`: `ready_for_stage_8`, Stage 01-08 passed, Stage 09 pending, keine blockierenden Issues.
- Creative OS Stage 7 ist gebaut und real fuer `creative-os-jungle-001` gelaufen: 3 passed Keyframes -> 3 LTX Motion Prompts -> Audit passed; kein LTX-Render gestartet.
- Aktueller enger Output: `/workspace/agent_runs/creative-os-jungle-001/creative_os/ltx_motion_prompts.json`, `ltx_prompt_audit.json`, `creative_os_stage7_report.md`.
- Naechster enger Schritt: Stage 8 als kontrollierten LTX I2V Render-Plan/Executor-Gate entwerfen, bevor irgendein Video-Render gestartet wird.
- Creative OS Stage 6 ist gebaut und real fuer `creative-os-jungle-001` gelaufen: Z-Image-Prompts -> 3 PNG-Keyframes -> heuristic QA -> Stage-6-Artefakte.
- Stage 6 bleibt bewusst vor LTX stehen. Aktueller enger Output: `/workspace/agent_runs/creative-os-jungle-001/creative_os/keyframes/` plus `keyframe_manifest.json`, `keyframe_review.json`, `keyframe_generation_log.json`, `creative_os_stage6_report.md`.
- Stage 6.1 hat `scene_02` manual-structured auf `passed` gesetzt; alle drei Keyframes sind fuer Stage 7 nutzbar.
- Creative OS V1 Dry-Run ist als andockbare Zusatzschicht gebaut. Aktueller enger Pfad: `scripts/creative_os_dry_run.py` -> `/workspace/agent_runs/<job_id>/creative_os/` -> 3 Z-Image-Keyframe-Prompts.
- Die Schicht ist nicht in `VideoAgent.run_job()` integriert und ersetzt keine bestehende Video-Pipeline. Sie dient als planender Prompt-Compiler bis `zimage_prompts.json`.
- Naechster enger Schritt: Creative OS V1 Output manuell gegen vorhandene `model_prompts.json`-/Prompt-Audit-Konventionen abgleichen, bevor ein Integrationspunkt in den bestehenden Stop-after-Pfad entschieden wird.
- Phase G2 ist umgesetzt: Die Content-Maschine hat jetzt eine Skill-Schicht unter `agent_core/creative_system/skills/`, eine skill-aware Pipeline `clean_shortform_v1`, Creative Role Contracts und initialen Decision-Log-Trace.
- Skills sind in G2 noch keine neue Executor-Engine. Sie sind ladbare, tracebare Wissensvertraege, die Pipeline Definitions, Prompt Audit und spaetere Director-/Planner-/Prompt-Builder-Entscheidungen strukturieren.
- Morning Reset nutzt jetzt flexible `motif_families` statt nur starrer Pflichtszenen. Die alten Shot Recipes bleiben als kompatible Bausteine erhalten.
- Neue CLI-Safety-Flags: `--pipeline-dry-run` und `--approval-gates-enabled` setzen nur Metadata, damit Operatoren Kontrolllaeufe explizit trocken starten koennen.
- G3/G4/G5 Final-Mega-Task ist umgesetzt: Stage Role Contracts sind tracebar, Stop-after ist im Core/CLI vorbereitet, Resume Contract ist dokumentiert/prüfbar, Creative-Quality-Warnungen und Decision Log sind erweitert.
- Aktueller sicherer Kontrollbefehl fuer morgen: `python3 scripts/agent_core_cli.py --idea "..." --script "..." --no-voice --pipeline-dry-run --stop-after model_prompts --approval-gates-enabled --print-payload`.
- Naechster sinnvoller Schritt nach G5: Skills und Stage Contracts aktiv in den Director/Planner einspeisen, sodass `clean_shortform_v1` echte Skill-gesteuerte kreative Entscheidungen trifft.
- Phase G1.1 ist umgesetzt: `--inspect-run` und `--inspect-checkpoints` zeigen Checkpoints; `--approve-checkpoint` und `--reject-checkpoint` schreiben lokale Gate-Dateien im Run-Ordner.
- G1.1 baut keinen Resume-Executor. Nach Freigabe zeigt die CLI klar, dass Resume vorbereitet ist, aber die eigentliche Fortsetzung des blockierten Executors als naechster Schritt definiert werden muss.
- Live-/Append-Ausgabe zeigt jetzt einen kleinen `CHECKPOINT`-Block, wenn `state.json` current/blocked Checkpoint-Felder enthaelt.
- Phase G1 ist umgesetzt: `simple_video_v1` beschreibt den bestehenden Video-Flow declarativ, `JobState` und `checkpoints.json` enthalten Checkpoints, und lokale Approval-Dateien koennen Plan-/Prompt-Gates blockierend freigeben.
- G1 bleibt bewusst ohne n8n-Integration, externe API-Erweiterung, GUI, Runtime-/Model-/Backend- oder `init.sh`-Aenderung.
- Fuer kontrollierte Trockenlaeufe: `job.metadata.pipeline_dry_run=true` verwenden. Dieser Modus stoppt nach Plan-/Prompt-Checkpoints und startet keine Voice-, Storyboard- oder Video-Backends.
- Fuer manuelle Gate-Pruefung: `job.metadata.approval_gates_enabled=true` setzen; dann wartet der Core an `approve_plan` bzw. spaeter `approve_prompts` auf `/workspace/agent_runs/<job_id>/approvals/<checkpoint_id>.json` mit `approved=true`.
- Naechster sinnvoller Schritt nach G1.1: Resume-Vertrag definieren und klein implementieren, sodass ein freigegebener blockierter Run kontrolliert weiterlaufen kann, weiterhin ohne n8n/API/GUI-Umbau.
- Tagesabschluss-F2 ist abgeschlossen: CLI Live Dashboard, Prompt Trace `model_prompts.json`, Backend Prompt Policy und F2 Creative OS Grundlage sind implementiert und getestet.
- Aktueller Dry-Run-Beleg: `/workspace/agent_runs/phase-f2-creative-os-dry-run`. Vor jedem echten Render zuerst `prompt_audit.json` und `model_prompts.json` pruefen.
- Backend-Prompt-Vertrag fuer Morning Reset: Z-Image positive-only, LTX positive + kurze Avoid-Liste. Keine Debuglabels oder Script-Snippets duerfen Backend-Prompts erreichen.
- Naechster Schritt morgen ist genau: Prompt-Audit/Model-Prompts des F2-Dry-Runs manuell pruefen; nur wenn sauber, `quality-morning-reset-009` manuell starten.
- Phase F1.1 Model Prompt Compiler Cleanup ist abgeschlossen. Naechster Pruefpunkt ist `/workspace/agent_runs/phase-f1-1-morning-reset-prompt-clean-dry-run/prompt_audit.json`.
- Erst wenn der F1.1-Audit manuell sauber wirkt, einen guenstigen echten `quality-morning-reset-009` starten.
- Phase F1 Creative Operating System + Prompt Audit ist abgeschlossen. Naechster Schritt: zuerst `/workspace/agent_runs/phase-f1-morning-reset-dry-run/prompt_audit.json` manuell anschauen.
- Erst nach Audit-Review einen echten `quality-morning-reset-009` starten; kein weiterer Mini-Fix ohne Audit-Beleg.
- Aktueller Architekturvertrag: Mode-/Style-Playbook -> Debug Prompt + Model Prompt -> Prompt Audit -> Z-Image/LTX nur mit model-facing Prompt.
- `quality-morning-reset-008` ist der naechste echte Testkandidat nach dem Narration-Isolation-Fix: subtitle-mode off, storyboard on, variations-per-scene 1, takes-per-scene 3, Qwen3-VL review.
- Der Fix nach `quality-morning-reset-007` ist abgeschlossen: deutsche Script-Snippets werden aus visuellen Prompts entfernt, in englische Visual Actions uebersetzt, Take-Review-Metadata vor Selection normalisiert und Device/UI-Risiken strenger bewertet.
- Dry-Run-Beleg liegt unter `/workspace/agent_runs/quality-morning-reset-008-plan-dry-run`; keine `Vorhang auf`/`Stell ein Glas Wasser ab`/`Atme ruhig am Fenster`/`Morning Reset:`-Leaks in visual prompts oder Storyboard effective prompts.
- `quality-morning-reset-007` ist der naechste echte Testkandidat nach dem Quality-Gate-Fix: subtitle-mode off, storyboard on, 3 takes per scene, Qwen3-VL review.
- Der naechste Fix nach `quality-morning-reset-006` ist abgeschlossen: rejected Take Selection verhindert, passed-score Konsistenz gehaertet, Keyframe Visual Gate gegen Text/Phone/Split-Screen erweitert, Qwen3-VL JSON robuster, Morning-Reset-Motive konkreter.
- Dry-Run-Beleg liegt unter `/workspace/agent_runs/quality-morning-reset-007-plan-dry-run`; alle selected keyframes sind `contract_preserved=true` und `visual_risk_status=passed`.
- Tagesabschluss/Backup ist der aktuelle Modus. Kein weiterer Feature-, Quality-, CLI-, Runtime- oder Init-Umbau in dieser Session.
- Morgen genau ein technischer Fokus: rejected selected Take Bug und Hard Keyframe Visual Gate gegen Text/Phone/Split-Screen fixen; danach neuer Clean-Visual-Test.
- Erster Morning-Reset-Qualitaetsfix nach Phase E2 ist abgeschlossen: Visual Prompt Sanitizer, Safe Morning Reset Motivbibliothek, allowed_props Cleanup, Storyboard-Prompt-Schutz und strengere Device-/UI-Review-Risiken.
- Naechster echter Test ist `quality-morning-reset-006` mit `--subtitle-mode off`, Storyboard, Qwen3-VL und 2-3 Takes pro Szene; danach visuell beurteilen statt weitere Runtime-Arbeit zu machen.
- Phase E2.2 CLI Dashboard Polishing ist abgeschlossen: keine doppelte Video-Zeile mehr, Vision Review klarer, Issues nach Quality/Vision/Config gruppiert, Subtitle-Burn-Hinweis und smartere Next Actions.
- Naechster fachlicher Schritt bleibt: `quality-morning-reset-005` visuell anschauen und danach gezielten Prompt-/Motiv-Fix machen, nicht Pipeline oder Runtime umbauen.
- Phase E2 CLI Dashboard / Produktionsansicht ist umgesetzt: Standardausgabe kompakter und dashboard-artig, `--inspect-run` nutzt die neue Summary, Fehlerausgabe zeigt Root Cause und Next Debug Command.
- Fuer Phase E2 wurden keine Pipeline-/Quality-/Prompt-/Model-/Director-/Backend-/Init-Aenderungen gemacht; es ist ein reiner CLI-UX-Schritt in `scripts/agent_core_cli.py` plus Doku.
- Naechster Schritt: echter Qualitaetsrun mit neuer CLI-Ausgabe und anschliessender visueller Analyse.
- Abschluss/Backup fuer heute: finaler Arbeitsstand wird schlank archiviert, ohne Modelle/Venvs/Caches; Qwen3-VL-Venv ist per Ensure-Script reproduzierbar.
- Nach Restore zuerst `HANDOFF.md` folgen. Der naechste fachliche Schritt ist echte Qualitaetsanalyse von `quality-morning-reset-003` und gezielter Motiv-/Prompt-Feinschliff.
- Dependency-Isolation ist abgeschlossen: LTX bleibt in der globalen FastAPI-Runtime auf `transformers 4.52.4`, Qwen3-VL laeuft in `/workspace/venvs/qwen3-vl-review` als Subprocess.
- `quality-morning-reset-003` ist der aktuelle technische Beleg: Director, Voice, Storyboard, LTX und Qwen3-VL-Review laufen zusammen; `final.mp4` wurde assembled.
- Naechster sinnvoller Schritt ist kein weiterer Dependency-Fix, sondern echte Qualitaetsruns ansehen/kalibrieren: Qwen3-VL meldete im Final Verdict sichtbare Subtitle-/Text-/Papier-Risiken.
- Qwen3-VL Runtime-Fix ist jetzt isoliert umgesetzt: nicht mehr FastAPI-/Worker-Python selbst, sondern die Qwen3-VL-Venv kennt `qwen3_vl` und FP8-Kernels; `evaluate_take_visual_review()` nutzt diese echte Inferenz per Subprocess.
- Kein weiterer Content-Maschine-Smoke wurde fuer diesen Fix gestartet; der naechste sinnvolle Lauf ist ein brauchbarer Morning-Reset-Qualitaetstest mit ruhiger Kueche/Fenster/Wasser/Bewegung und den CLI-Vision-Flags.
- Vision-Review-Provider-Wiring ist umgesetzt: neue CLI-Flags schreiben Vision-Review-Settings in Job-Metadata, und Agent/Utils bevorzugen diese Metadata vor Env.
- Naechster konkreter Test: `readiness-storyboard-vision-003` mit `--vision-review-enabled --vision-review-provider qwen3_vl --vision-review-model-dir /workspace/models/Qwen3-VL-4B-Instruct-FP8 --vision-review-max-frames 3`.
- Phase E CLI Produktions-Cockpit ist umgesetzt: bessere Live-Ausgabe, Director-/Step-/Take-Summary, Quality Verdict und strukturierte Failure-Diagnose inklusive Backend-`job.log`-Tail.
- Neue CLI-Diagnoseoptionen sind verfuegbar: `--inspect-run`, `--tail-error-log-lines`, `--no-log-tail`, `--quiet`, `--verbose`.
- Naechster Schritt ist ein echter kleiner Storyboard-/Vision-Review-Test mit der neuen CLI-Ausgabe und danach Qualitaetsfeinschliff.
- LTX/Gemma-Readiness ist repariert: Gemma wurde index-vollstaendig nachgeladen und `init.sh` prueft Gemma jetzt ueber Tokenizer, Preprocessor, Index und Shards statt nur `config.json`.
- `readiness-small-social-003` ist der aktuelle gruene End-to-End-Beleg: Director `llm_augmented`, Voice, LTX und muxed `final.mp4` erfolgreich.
- Naechster Schritt bleibt ein echter kleiner Video-/Qualitaetscheck oder Phase E CLI Produktions-Cockpit, nicht weitere Init-/Modellreparatur.
- Finaler Init-Fokus 2026-04-30: Morgen nach frischem Pod soll `bash /workspace/init.sh` vorhandene Modelle skippen, fehlende Modelle laden und parallele Init-Laeufe per kleinem `flock` verhindern.
- `hf_transfer` ist nicht mehr Default; stabiler Init-Default ist `HF_HUB_ENABLE_HF_TRANSFER=0`, Xet bleibt aus. Optionaler Speed nur bewusst per `HF_HUB_ENABLE_HF_TRANSFER=1 bash /workspace/init.sh`.
- Qwen3-VL bleibt optional ueber `Qwen3_VL_Review=on` oder `Vision_Review_Model=on`; kein Qwen3-VL-Adapter-, `agent_core`-, API-, llama.cpp- oder Phase-E-Umbau in diesem Init-Schritt.
- Naechster Projekt-Schritt bleibt Phase E CLI Produktions-Cockpit oder ein echter kleiner Video-Test.
- Aktueller Init-Fokus 2026-04-30: `/workspace/init.sh` ist wieder die kleine OG-basierte Init, nicht die grosse Guard-/Lock-/Heartbeat-Version.
- Der Director-Autostart-Fix bleibt minimal enthalten: vorhandenes `serve_director_llm.sh` wird best-effort executable gemacht und per `DIRECTOR_LLM_DAEMON=1 bash ...` gestartet.
- Qwen3-VL ist im Init nur optional verdrahtet: `tools.config` enthaelt den sichtbaren Schalter `Qwen3_VL_Review`, Aktivierung ueber `Qwen3_VL_Review=on` oder `Vision_Review_Model=on`.
- Qwen3-VL-Download/Verify ist ausgelagert nach `/workspace/scripts/download_qwen3_vl_model.py`; kein neuer Init-Modus, kein Phase-E-Bau, kein `agent_core`-/API-/llama.cpp-Umbau in diesem Schritt.
- Qwen3-VL-Modellsetup ist abgeschlossen: `Qwen/Qwen3-VL-4B-Instruct-FP8` liegt unter `/workspace/models/Qwen3-VL-4B-Instruct-FP8`, Dateien und CPU-Load-Smoke sind verifiziert.
- Qwen3-VL echter Bild-Smoke ist jetzt ebenfalls verifiziert: kleines lokales Testbild, `provider=qwen3_vl`, `take_visual_review_status=passed`, `postability_score=1.0`.
- Phase C ist jetzt umgesetzt: Take Visual Review / Postability Score laeuft nach technischer Take-Validation, extrahiert Review-Frames, bewertet heuristisch gegen Scene World Contract und priorisiert Take-Auswahl visuell.
- Optionaler Qwen3-VL Provider ist lazy vorbereitet; Default bleibt `heuristic`, kein Init-/Startup-Hook und keine VLM-Pflicht fuer normale Tests.
- Phase D ist jetzt umgesetzt: Final Quality Verdict landet in `ResultSummary.metadata.final_quality_verdict` und kombiniert technische Final-MP4-Validation, Assembly-, Take-, Keyframe-, Subtitle-/Overlay-, Voice-/Music- und Final-Frame-Quellen.
- Einschub 2026-04-29: Init-/Download-/Startup-Pfad wurde gezielt gehaertet, weil der Pod real im Z-Image-Turbo-Download hing. Dieser Einschub hat keinen Phase-C-Bau gestartet und keine Pipeline-/agent_core-/Runtime-Architektur geaendert.
- Der naechste Feature-Schritt ist nach Phase D jetzt Phase E; aktueller technischer Stand aus dem Init-Einschub bleibt: Init blockiert nicht mehr still, parallele Init-Laeufe werden verhindert, unvollstaendige HF-Snapshots werden nicht mehr als fertig akzeptiert.
- Folgeabschluss 2026-04-29: Director-Download-/Startup-Pfad ist jetzt real gruen. Das konfigurierte Qwen3.6-35B-A3B-GGUF liegt lokal, `llama-server` laeuft auf `127.0.0.1:8011`, `/v1/models` und `scripts/check_director_llm.py` sind gruen. Kein Phase-C- oder Qwen3-VL-Start.
- Aktueller Output-Quality-Fokus Phase A ist umgesetzt: Scene World Contract + PromptBuilder v2 haerten Szene- und Variation-Prompts gegen Text-/Screen-/Papier-Artefakte, ohne Runtime-/Backend-Aenderung.
- Phase B1 ist umgesetzt: Storyboard-/Keyframe-Kandidaten erhalten jetzt scene-specific, contract-aware `effective_prompt`-Metadaten; Z-Image nutzt diese bevorzugt statt nur globaler Plan-Prompts.
- Phase B2 ist umgesetzt: Storyboard-Keyframe-Kandidaten erhalten jetzt einen leichten `visual_risk_review` mit Status `passed`, `needs_review` oder `rejected`; Auswahl bevorzugt `passed` vor `needs_review` vor `rejected`.
- Phase B1/B2 sind per Unit-Tests und Dry-Run-Artefaktplaenen verifiziert; noch kein neuer langer GPU-Render und keine finale visuelle Qualitaetsbehauptung.
- Naechster Output-Quality-/Produktionsschritt: Phase E CLI Produktions-Cockpit.
- Danach: echte Video-Qualitaetstests und Qualitaetsfeinschliff.
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
# Aktiver Plan: Creative OS CLI Cockpit V1.6 Abschluss

Creative OS ist bewusst vor Stage 8 gestoppt. Der aktuelle Abschlussstand ist ein read-only CLI Cockpit:
- `scripts/creative_os_status.py --style plain` bleibt stabil.
- `scripts/creative_os_status.py --style rich` zeigt V1.6 Rich Cockpit Grid.
- Snapshots liegen unter `/workspace/cli_cockpit_snapshots/`.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 12 Tests OK.
- Nicht gebaut: Stage 8, Render, LTX, Video, Backend-Aufruf, n8n, API, neue Creative-OS-Stages, Textual.

Naechster enger Schritt: Design visuell vom Operator pruefen.

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
# Aktiver Plan: G10 Content Maschine V1 Tuning / Seele

G9 ist abgeschlossen. Der erste kontrollierte echte V1-Run lief end-to-end mit G6/G7/G8-Trace, Storyboard, LTX, Final Quality Verdict, FeedbackActions und RetryPlan. Der Clip ist technisch erfolgreich, aber wegen sichtbarer Text-/UI-/Papierartefakte in Szene 2 nicht demo-wuerdig.

Der naechste sinnvolle Schritt ist G10 Content Maschine V1 Tuning / Seele:
- Szene-2-Motiv/Shot-Recipe vom dokument-/papierartigen Objekt wegfuehren.
- Taktile Motive staerker physisch und weniger UI-/Papier-assoziiert machen.
- G8 FeedbackActions als manuelle Tuning-Anleitung nutzen, weiterhin ohne Auto-Retry.
- Keine Runtime-, Modell-, Docker-, `init.sh`-, Backend-, n8n/API/GUI-Umbauten ohne neuen expliziten Auftrag.
