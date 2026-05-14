# CHANGELOG.md

## 2026-05-14 Phase 1 Live Orchestrator V3 Fix
- Phase-1-Grenze in der Pipeline Map erzwungen: Stage `10` bis `15` bleiben fuer Phase-1-Runs pending/out-of-scope und koennen nicht durch vorhandene Artefakte gruen werden.
- Stage `09` Active Workspace zeigt echte Manifest-/Live-Daten sichtbar: Backend Status/Reason, Overall Status, Live Stage 09, Finished Files und Gallery.
- Stage `09` Manifest wird waehrend echter Image-Generierung fortgeschrieben, damit Watch running/progress/finished/error aus realem State lesen kann.
- Watch-Refresh vergleicht semantischen State ohne `last_refresh_time`; reine Timer-Ticks loesen keinen Full-Panel-Refresh aus und manuelle Stage-Auswahl bleibt erhalten.
- Tests erweitert fuer Stage-10+-Grenze, Stage-09-Manifestanzeige, Watch-Auswahl und bestehende Kompatibilitaet.
- Smokes: `live-v3-smoke-noimages-20260514` und `live-v3-smoke-images-20260514`.

## 2026-05-14 Phase 1 Live Orchestrator V2 Fix
- Status-Konsistenz korrigiert: disabled/missing Backend schliesst nur Stage `00` bis `08` ab; Stage `09` bleibt `error` und Gesamtstatus `paused_missing_backend`.
- `stage_events.jsonl` schreibt bei failed/disabled Keyframe-Jobs kein `09 done` mehr.
- Pipeline Map priorisiert `live_status.stages[*].status`; Stage `09` wird bei paused/missing Backend nicht mehr durch vorhandenes `keyframe_manifest.json` gruen.
- `--open-cockpit` entschärft: kein Textual-Background-Prozess im selben TTY, stattdessen klare Zwei-Terminal-Anleitung ohne OSError.
- Neue CLI-Optionen: `--stage-delay-seconds`, `--generate-images`, `--no-generate-images`; `--no-images` bleibt kompatibel.
- Smokes: `live-v2-smoke-20260514` disabled mit Stage `09=error`; `live-v2-smoke-images-20260514` mit Backend, 3 PNGs und Gallery.

## 2026-05-14 Phase 1 Live Cockpit Orchestration
- `creative-os run-phase1-live` als zusaetzlicher CLI-Befehl eingefuehrt; `creative-os run-phase1` bleibt Batch-kompatibel.
- Neuer Live-State unter `creative_os/live_status.json` plus `stage_events.jsonl` mit Stage-Status `pending -> running -> done/error/missing`, Timestamps, Artifact-Pfaden, `viewed_stage`, `real_run_stage` und `current_running_stage`.
- Phase-1-Live-Runner schreibt Stage `00` bis `09` sequenziell: normalized job, pipeline route, mode/style, skill tree, strategy, beat/hook plan, judge, scene contracts, prompt payload und keyframe manifest.
- Textual Cockpit Watch liest Live-State, startet Live-Runs bei Stage `00`, markiert den echten Runner-Stage separat und haelt Stage `09`/complete nach Finish.
- Real-Run Stage `09` zeigt keine Fake-Cards und keinen geratenen Progress; fehlendes Manifest bleibt `missing manifest`, finished PNGs nutzen echte Datei-Metadaten.
- Verifikation: Live-Smoke `live-smoke-20260514`; Status-Tests 23 gruen, Cockpit-Tests 16 gruen. Keine Stage `10-15` Runtime, kein LTX, kein n8n/API, kein Redesign, Textual bleibt `0.89.x`.

## 2026-05-09 Cockpit Panel Completion V0.2
- Active Workspace Stage-Panels 00-15 als read-only Stage-Oberflaechen vervollstaendigt: 04-08 und 10-15 zeigen jetzt Stage-spezifische Status-, Artifact-, Expected-Output- und Next-Action-Informationen statt generischer Placeholder.
- Stage 00-03 bleiben als Command Center, Pipeline wählen, Mode & Style und Skills laden ausgearbeitet; Stage 00 enthaelt jetzt einen read-only Command Composer mit sichtbaren Topic/Format/Mode/Style/Duration/Voice/Music/Subtitles/Storyboard/Output-Feldern, Command Preview und deaktiviertem V0.2-Run-Hinweis.
- Stage 09 Image Jobs bleibt im bestehenden Card-Look mit Preview-Slots, Expanded Image 2 und Unicode-Progressbar erhalten; V0.2-Snapshots liegen unter `/workspace/cockpit_snapshots_2026-05-09_v02/`.
- Keine Render-, API-, n8n-, Run-, Command-Execution- oder Pipeline-Integration und kein globales Redesign.

## 2026-05-09 Textual Cockpit Stage 09 Image Jobs Pass
- Active Workspace Stage 09 erweitert: `PROMPTS / IMAGE JOBS` zeigt jetzt drei einzelne Image-Job-Bloecke mit Preview-Zone, Job/Prompt-Zeile, Status und Expand-Pfeil.
- Lokale Bedienung vorbereitet: j/k waehlt Image Jobs in Stage 09, Enter/Space klappt den selektierten Job auf/zu; Pfeil hoch/runter bleibt fuer Stage-Auswahl.
- Fixture-Demo zeigt sauber gekennzeichnete Job-Zustaende: Image 1 ready, Image 2 generating mit Demo-Progress/%/elapsed/backend, Image 3 in queue.
- Kein globales Layout, keine Sidebar, keine Header-/Bottom-Panel-Aenderung, keine Pipeline-Integration, kein Render und keine API-/n8n-Arbeit.

## 2026-05-09 Textual Cockpit Stage Router V0.1
- Stage-Cockpit V0.1 eingebaut: Pipeline Map V1 zeigt die Operator-Stages 00-15 von Command Center bis Final Output.
- Pipeline Map ist auswählbar: Tastatur-Fallback ueber Pfeil hoch/runter sowie j/k, selected Stage ist sichtbar markiert; Klick-Handler fuer Pipeline-Map-Auswahl ist vorbereitet.
- Active Workspace routet anhand der selected Stage: 00-03 haben initiale read-only Views, 04-15 funktionale Platzhalter; Stage 09 nutzt weiter das bestehende Prompts/Image-Jobs-Panel.
- Keine Live-Pipeline-Integration, keine echte CLI-Eingabe, kein Job-Submit, kein Render, keine API-/n8n-Aufrufe und keine Fake-Livewerte.

## 2026-05-08 Textual Cockpit Header / Active Workspace Reference Pass
- Header oben naeher an Kommandozentrale-Struktur gebracht: linke CM-Brand-/Titelzone, mittlere Command-Banner-Metadaten und rechte Time/Session/Operator/Run-Type/Watch-Spalte.
- Active Workspace oben in klare innere Subbereiche gegliedert: `CURRENT POSITION AND PIPELINE PATH`, `PROMPTS / IMAGE JOBS` und `PIPELINE FLOW`.
- Alte Scene-Card-Liste im Active Workspace durch kompakte Prompt-/Image-Job-Zeilen ersetzt; Texte werden gekuerzt, Source/Keyframe und Status bleiben sichtbar.
- Pipeline Path und Flow sind statisch/read-only aus Run-Typ und vorhandenen Artefaktdaten abgeleitet; keine Fake-Livechecks, keine ETA, keine Prozentwerte.
- Keine Render-, LTX-, API-, n8n- oder Pipeline-Integration und keine neuen Runs.

## 2026-05-08 Textual Cockpit Active Workspace Detail Pass
- Active Workspace im bestehenden Textual Cockpit verdichtet: zweispaltige Status-Zone mit Current Step, Last Passed, Next Technical, Operator Focus, Render Paused, Run Type, Final MP4 und Director Mode.
- Stage Output zeigt jetzt read-only kompakte Run-Ausgaben fuer Creative-OS-Fixtures und Agent-Core-Runs, inklusive Scene Count, final.mp4 Status, stop_after und director_mode falls vorhanden.
- Scene Cards bleiben im bestehenden Look, zeigen fuer Agent-Core-Runs read-only Scene-Plan-/Model-Prompt-Zusammenfassungen und gekuerzte Texte innerhalb der festen Card-Breite.
- Run Notes ergaenzt: Session, Watch-Status und Hinweis auf no live backend calls/read-only view.
- Keine Pipeline-Integration, keine API-/n8n-Integration, kein Render und keine neuen Runs.

## 2026-05-07 Final Closeout Cockpit Semantics + Video Proof
- Textual Cockpit Issue-Semantik verengt: Director-/Runtime-Probleme und `final.mp4`-Status bleiben im Issues/Next-Kontext; Skill Health zeigt nur Skill-System-Zustand.
- Issues Panel hat jetzt abgeleitete Severity `none|warning|error` und passende Border-Farben: cyan/blau fuer keine Issues, amber fuer Fallback/Degraded, rot fuer Blocking/Error.
- Finaler Video-Smoke-Run `cockpit-video-smoke-001` lief ohne `--stop-after` ueber den bestehenden Agent-Core-CLI/API-Pfad: `director_mode=llm_augmented`, `director_llm_active=true`, `final.mp4` erzeugt.
- Cockpit V0.4 Watch gegen `cockpit-video-smoke-001` verifiziert: `run_type=agent_core`, Watch on, Director llm_augmented, `final.mp4 present`, keine Director-Fallback-Issue, Skill Health sauber.
- Kein Redesign, keine neuen grossen Features, keine n8n-/Settings-/Pipeline-Integration und kein Render-Ausbau.

## 2026-05-07 Director 8011 Restore + Cockpit Smoke
- Director 8011 war vorher down; Ursache war ein Restore-/Runtime-Zustand der vorhandenen `llama.cpp`-Installation: `llama-server` war nicht ausfuehrbar und lokale `.so.0`-Symlinks fehlten fuer den Runtime-Library-Pfad.
- Minimalfix ueber bestehenden Pfad: `bash /workspace/scripts/ensure_llama_cpp.sh`, danach `bash /workspace/init.sh`; kein Rebuild und keine Cockpit-Codeaenderung.
- `check_director_llm.py` meldet wieder `director_llm_active=true` gegen `http://127.0.0.1:8011/v1/chat/completions`.
- Neuer kontrollierter Agent-Core-Smoke-Run: `/workspace/agent_runs/cockpit-realrun-smoke-002`, `director_mode=llm_augmented`, `director_llm_active=true`, Stop nach `model_prompts`, kein `final.mp4`.
- Cockpit V0.4 liest den neuen Run read-only/watch als `agent_core`; kein Director-Fallback-Issue sichtbar, `final.mp4 missing` bleibt korrekt sichtbar.

## 2026-05-07 Textual Cockpit V0.4
- Textual Cockpit State Adapter erkennt jetzt read-only Agent-Core-Runs direkt unter `/workspace/agent_runs/<job-id>` zusaetzlich zu Creative-OS-Runs unter `creative_os/`.
- Run-Typen werden unterschieden: `creative_os`, `agent_core`, `missing`, `unknown`; Header zeigt den Run Type ohne Layout-Redesign.
- Agent-Core-Artefakte werden defensiv gemappt: `result.json`, `state.json`, `plan.json`, `scene_plan.json`, `model_prompts.json`, `prompt_audit.json`, `director_output.json`, `final.mp4`.
- `rule_based_fallback`, `director_llm_active=false` und fehlendes `final.mp4` erscheinen als Issues/Next-Hinweise.
- Creative-OS-Fixture bleibt kompatibel; `/workspace/agent_runs` bleibt read-only. Keine Pipeline-Integration, kein Render, kein LTX, kein Video, kein Director-Fix.

## 2026-05-07 Textual Cockpit V0.3
- Real-Run-Readiness ergaenzt: `scripts/creative_os_cockpit.py --job-id <job-id>` liest defaultmaessig read-only aus `/workspace/agent_runs/<job-id>`.
- Fixture/Demo-Modus bleibt ueber `--runs-root /workspace/tests/fixtures/creative_os_runs` erhalten und wird im Header weiter als `fixture/demo` angezeigt.
- Missing-Run-Zustand zeigt ohne Traceback `Run not found`, den gesuchten Pfad und einen Hinweis auf `--runs-root` oder das Erzeugen eines echten Runs.
- Read-only Watch Mode ergaenzt: `--watch --refresh-sec 2` laedt die Inspector-Daten periodisch neu; `r` bleibt manueller Refresh, `q` beendet.
- `/workspace/agent_runs` bleibt fluechtige Datenquelle und wird vom Cockpit nicht beschrieben.
- Keine Pipeline-Integration, kein Stage 8, kein Render, kein LTX, kein Video, keine API, kein n8n, keine Settings-UI.

## 2026-05-07 Textual Cockpit Panel V0.2
- Modulare Textual-Cockpit-Panel-Struktur eingefuehrt unter `agent_core/creative_os/cockpit/`.
- Panels sind jetzt getrennte Module: Header, System Status, Pipeline Map, Active Workspace, Skill Health, Artifacts, Issues und Next.
- Registry/Config vorbereitet: `panel_registry.py` enthaelt Panel-Metadaten, Default-Regionen und `PANEL_CONFIG` fuer spaeteres Enable/Region-Verhalten.
- State Adapter eingefuehrt: `state_adapter.py` wandelt Fixture-/Inspector-Daten in panelnahe Datenstrukturen um.
- Sichtbarer Cockpit-Look bleibt beibehalten; `agent_core/creative_os/textual_cockpit.py` ist eine Kompatibilitaetsschicht.
- Keine echte Pipeline-Integration, kein Stage 8, kein Render, kein LTX, kein Video, keine API, kein n8n.

## 2026-05-06 Creative OS Textual Cockpit V0.1
- Neuer Textual-TUI-Prototyp gebaut: `agent_core/creative_os/textual_cockpit.py` und Entry `scripts/creative_os_cockpit.py`.
- Startbefehl: `python3 /workspace/scripts/creative_os_cockpit.py --job-id creative-os-jungle-001 --runs-root /workspace/tests/fixtures/creative_os_runs`.
- Layout: Fullscreen-Cockpit mit Header, linker Sidebar fuer System/Pipeline, grossem Active Workspace, Scene Cards, Bottom-Kacheln und Footer-Keybinds `q`, `r`, `h`.
- Datenquelle bleibt der vorhandene `CreativeOSRunInspector`; Refresh liest nur neu ein und schreibt keine Run-Artefakte.
- `textual>=0.89,<1.0` ist in `requirements.txt` dokumentiert; das Script meldet fehlendes Textual mit Installationshinweis.
- `/workspace/agent_runs` bleibt fluechtig und wurde nicht als Design-/Testquelle verwendet oder mutiert.
- Nicht gebaut/gestartet: Stage 8, Render, LTX, Video, API, n8n, Backend-Aufrufe oder neue Creative-OS-Stages.

## 2026-05-06 Creative OS Rich Cockpit V1.7
- Rich-Cockpit-Design-Pass V1.7 umgesetzt: Theme-System weiter zentralisiert, Header kompakter gemacht, Fixture/Demo-Session im Banner sichtbar, Stage 09 als naechster Schritt staerker akzentuiert und Active Workspace um einen kompakten technischen Flow-Streifen ergaenzt.
- Alle UI-Checks und Snapshots laufen explizit mit `--runs-root /workspace/tests/fixtures/creative_os_runs`; `/workspace/agent_runs` bleibt fluechtig und wird nicht als Design-/Testquelle genutzt.
- `--style plain` bleibt stabil; `--view all --style rich` zeigt zuerst das Cockpit-Overview und danach getrennte Detailbereiche.
- Snapshots aktualisiert unter `/workspace/cli_cockpit_snapshots/`.
- Nicht gebaut/gestartet: Stage 8, Render, LTX, Video, API, n8n, Textual, Backend-Livechecks oder neue Creative-OS-Stages.

## 2026-05-06 Creative OS CLI Cockpit Design-Pass
- Rich-Overview des read-only Creative-OS-Status-Dashboards als Operator-Cockpit ueberarbeitet: zentrales Theme, kompaktes Status-Banner, Sidebar/Workspace-Hauptgrid, Scene-Job-Karten und sauberes Bottom-Grid.
- `--style plain` bleibt stabil; `--style rich` ist der neue Cockpit-Modus. `--view all --style rich` zeigt zuerst das Overview und danach getrennte Detailbereiche.
- Run-Root-Abhaengigkeit geklaert: `/workspace/agent_runs` ist nur der Default-Ort fuer fluechtige echte Run-Artefakte, keine Systemquelle, kein Config-Ort, kein Skill-Ort und keine Pflichtabhaengigkeit.
- `scripts/creative_os_status.py` akzeptiert `--runs-root`; Design-/Testdaten liegen stabil unter `/workspace/tests/fixtures/creative_os_runs`.
- Fehlende Runs melden jetzt explizit, dass dies kein Systemfehler ist und dass nach Pod-Reset echte Runs neu erzeugt oder per `--runs-root` Fixtures gelesen werden muessen.
- Keine Stage 8, kein Render, kein LTX-Lauf, kein Video, keine API-, n8n-, Backend- oder Textual-Integration gebaut.
- Snapshots erzeugt unter `/workspace/cli_cockpit_snapshots/` gegen die isolierte CLI-Fixture, weil der echte Run-Ordner `/workspace/agent_runs/creative-os-jungle-001/creative_os` im aktuellen Workspace fehlt.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 13 Tests OK.

## 2026-05-05 Creative OS CLI Dashboard V1
- Read-only CLI-Kommandozentrale gebaut: `agent_core/creative_os/run_inspector.py`, `dashboard.py` und `scripts/creative_os_status.py`.
- Views: `overview`, `skills`, `stages`, `artifacts`, `issues`, `next`, `all`.
- Status wird ausschliesslich aus vorhandenen Creative-OS-Artefakten abgeleitet; API/Director bleiben `? not_checked`, wenn sie nicht real geprueft werden.
- Beleglauf: `python3 /workspace/scripts/creative_os_status.py --job-id creative-os-jungle-001 --view overview` und `--view all` liefen mit Exit-Code 0.
- Dashboard erkennt fuer `creative-os-jungle-001`: Stage 01-08 passed, Stage 09 pending, `ready_for_stage_8`, keine blockierenden Issues.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 5 Tests OK.
- Bewusst nicht gebaut: LTX Render, Video, Backend-Aufrufe, Qwen-VL-Aufruf, n8n, API, Batch, Mutationen der Run-Artefakte oder neue Creative-OS-Stages.

## 2026-05-05 Creative OS Stage 7 LTX Motion Prompt Compiler
- Creative OS Stage 7 gebaut: `agent_core/creative_os/ltx_motion_prompt_compiler.py`, `stage7_runner.py` und `scripts/creative_os_ltx_prompts.py`.
- Stage 7 nutzt nur passed Keyframes plus Scene Contracts, Keyframe Contracts, Creative Strategy, Z-Image-Prompts und Keyframe Reviews.
- Realer Lauf: `python3 /workspace/scripts/creative_os_ltx_prompts.py --job-id creative-os-jungle-001`.
- Ergebnis: 3 LTX-I2V-Motion-Prompts in `ltx_motion_prompts.json`; Audit `ltx_prompt_audit.json` overall `passed`, alle Szenen `passed`, `render_started=false`.
- Report geschrieben: `creative_os_stage7_report.md`.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_ltx_prompts.py -v` -> 2 Tests OK.
- Bewusst nicht gebaut: LTX Render, Video, Take Review, Final Assembly, n8n, API-Erweiterung, Batch-System, grosse Retry-Loops oder Hauptpipeline-Refactor.

## 2026-05-05 Creative OS Stage 6 Keyframes + QA
- Creative OS Stage 6 gebaut: `agent_core/creative_os/keyframe_generator.py`, `keyframe_qa.py`, `stage6_runner.py` und `scripts/creative_os_keyframes.py`.
- Stage 6 liest vorhandene Creative-OS-`zimage_prompts.json`, generiert Keyframes ueber den bestehenden Z-Image-HTTP-Pfad und schreibt Manifest, Generation Log, Review und Stage-6-Report.
- Realer Lauf: `python3 /workspace/scripts/creative_os_keyframes.py --job-id creative-os-jungle-001 --review-provider heuristic --max-wait-sec 900`.
- Ergebnis: echte PNGs erzeugt unter `/workspace/agent_runs/creative-os-jungle-001/creative_os/keyframes/scene_01.png`, `scene_02.png`, `scene_03.png`.
- QA: heuristic file/prompt review; `scene_01=passed`, `scene_02=needs_review` initial, danach Stage 6.1 manual-structured `passed`, `scene_03=passed`.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_keyframes.py /workspace/tests/test_creative_os_keyframe_qa.py -v` -> 6 Tests OK.
- Bewusst nicht gebaut: LTX Motion Prompt Compiler, LTX Render, Video, n8n, API-Erweiterung, Batch-System, grosse Retry-Loops oder Refactor der Hauptpipeline.

## 2026-05-05 Creative OS V1 Dry-Run Layer
- Isolierten Zusatzpfad `agent_core/creative_os/` gebaut: Schemas, Skill Loader, Intent Router, Creative Strategist, Beat Planner, Creative Judge, Scene Contracts, Keyframe Contracts, Z-Image Prompt Compiler und Runner.
- Neues Script `scripts/creative_os_dry_run.py` erzeugt aus einem Job bis zu 3 Z-Image-Keyframe-Prompts und stoppt dort.
- Pflichtartefakte werden unter `/workspace/agent_runs/<job_id>/creative_os/` geschrieben: normalized job, intent route, skill match, strategy, candidates, selected plan, scene contracts, keyframe contracts, zimage prompts, decision log und Markdown-Report.
- Kleine reale Skillbibliothek angelegt; fehlender `modes/jungle_adventure` crasht nicht, sondern nutzt `fallback/generic_visual_adventure` plus `styles/cinematic_nature`.
- Beleglauf `creative-os-jungle-001` erzeugte 3 positive, einzelne Keyframe-Prompts ohne Render.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_skill_loader.py /workspace/tests/test_creative_os_runner.py /workspace/tests/test_creative_os_prompts.py -v` -> 4 Tests OK.
- Bewusst nicht gebaut: Bildrender, LTX-Render, Qwen-VL-Zwang, Batch, n8n, API-Erweiterung, Runtime-/llama.cpp-Umbau und Integration in den bestehenden Video-Executor.

## 2026-05-02 Final Mega Task G3/G4/G5 Architecture + Archive Prep
- G2 wurde auditiert: Skills, Loader, `simple_video_v1`, `clean_shortform_v1`, Skill Trace, Prompt Policies und CLI Safety Flags sind vorhanden und getestet.
- G3 umgesetzt: `agent_core/creative_system/contracts.py` erzeugt Stage Role Contracts fuer `CreativeStrategy`, `BeatPlan`, `VisualDirection`, `ModelPromptPlan` und `ReviewPlan`.
- `stage_contracts.json` wird pro Run geschrieben und in `prompt_audit.json`/`model_prompts.json` gespiegelt.
- G4 umgesetzt: CLI/Core akzeptieren `--stop-after scene_plan|model_prompts|storyboard` bzw. `job.metadata.stop_after`; Stop-after-Resultate markieren `stopped_after`, `produced_artifacts`, `next_action`, `render_started=false` und `model_backends_started=false`.
- G4 Resume bleibt bewusst Future Work als Executor; `agent_core/resume_contract.py` beschreibt und prueft den Resume-Vertrag, Rejections und wiederverwendbare Artefakte.
- G5 umgesetzt: `evaluate_creative_quality_metadata()` liefert metadata-basierte Warnungen fuer boring/no-action/static/generic-stock/composition/platform-fit Risiken ohne Fake-VLM-Behauptung.
- `evaluate_final_quality_verdict()` akzeptiert jetzt `creative_quality_warnings` und `platform_fit_warnings` aus Take Reviews.
- Qwen3-VL Reviewer-Systemprompt wurde um kreative Qualitaetskriterien erweitert, ohne den JSON-only Vertrag aufzugeben.
- Decision Log erweitert um `approval_gate_status`, `stop_after` und `quality_decision`-Contract.
- Sicherer In-Process-Smoke ohne Render: `/workspace/agent_runs/g5-final-stop-after-model-prompts-smoke`.
- Keine Render, keine Modellladung, keine Downloads, keine Runtime-/Dependency-/Docker-/init.sh-/Backend-/n8n-/API-/GUI-Aenderungen.

## 2026-05-02 Phase G2 Skill Layer + Pipeline Modes + Creative Roles
- Neue Skill-Struktur unter `agent_core/creative_system/skills/` fuer Models, Platforms, Stages, Directing, Prompting und Review.
- Jede Skill-Datei ist Markdown mit `title`, `purpose`, `when_to_use`, `rules`, `do`, `dont`, `output_contract`, `common_failures` und `audit_hints`.
- Neuer Loader `agent_core/creative_system/skill_loader.py`: `load_skill`, `load_required_skills`, `resolve_skills_for_pipeline`; fehlende Skills werden als `missing_skills` gemeldet statt hart zu crashen.
- `PipelineDefinition` und `PipelineStepDefinition` koennen jetzt `required_skills` deklarieren; `PipelineDefinition` kann `stage_roles` dokumentieren.
- Neue Pipeline `agent_core/pipeline_defs/clean_shortform_v1.json` fuer kurze Social-Videos mit Skill-Anforderungen fuer Creative Strategy, Beat Planning, Visual Direction, Model Prompting, Z-Image, LTX und Review.
- Neue Creative Role Contracts in `agent_core/schemas.py`: `CreativeStrategy`, `BeatPlan`, `VisualDirection`, `ModelPromptPlan`, `ReviewPlan`; zusaetzlich `DecisionLog`/`DecisionLogEntry`.
- `VideoAgent` resolved Pipeline-/Mode-/Style-Skills nach dem Planaufbau und schreibt `required_skills`, `loaded_skills`, `missing_skills`, `stage_roles` und `motif_families` in Plan-Metadata.
- `prompt_audit.json` und `model_prompts.json` enthalten jetzt Skill Trace, Pipeline-ID, Backend-Prompt-Policy-Notizen sowie `ltx_positive_prompt_sent`/`ltx_negative_prompt_sent`-Tracefelder.
- `decision_log.json` wird als neues Run-Artefakt vorbereitet und nach dem Planen geschrieben.
- Morning Reset Mode fuehrt flexible `motif_families` und `motif_family_guidance` ein; bestehende Scene-Arc-Rezepte bleiben als kompatible empfohlene Bausteine.
- CLI-Safety-Flags ergaenzt: `--pipeline-dry-run` und `--approval-gates-enabled` setzen nur Job-Metadata.
- Tests: G2-Tests, G1/G1.1-Tests, Creative/Planner-Tests und Compileall wurden verifiziert.
- Kein Render, keine Modellladung, keine Downloads, keine Runtime-/Docker-/init.sh-/Backend-/n8n-/API-/GUI-Aenderungen.

## 2026-05-02 Phase G1.1 CLI Checkpoint Inspect + Approval/Reject UX
- `scripts/agent_core_cli.py --inspect-run <job>` zeigt jetzt zusaetzlich Checkpoints an, gelesen aus `checkpoints.json` oder fallback aus `state.json.checkpoints`.
- Neuer Checkpoint-Inspect-Modus: `--inspect-checkpoints <job_id_or_path>`.
- Neue lokale Gate-Befehle:
  - `--approve-checkpoint <job_id_or_path> <checkpoint_id> --approved-by "human" --approval-note "..."`
  - `--reject-checkpoint <job_id_or_path> <checkpoint_id> --rejected-by "human" --approval-note "..."`
- Die CLI schreibt `approvals/<checkpoint_id>.json` im Run-Ordner mit `approved`, `approved_by`, `approved_at` und `note`.
- Sicherheitsregeln: Checkpoint muss existieren; Approval-Dateien werden nicht ueberschrieben ausser mit `--force-approval`; Pfad-Escape aus dem Run-Ordner wird abgelehnt.
- Live-/Append-Dashboard zeigt einen kleinen `CHECKPOINT`-Block mit current/blocked Checkpoint, Status, Approval-Pflicht und Next-Action-Hinweis.
- Resume ist bewusst nur vorbereitet: Die CLI zeigt fehlende Approval-Datei und Befehl, aber ein echter Resume-Executor bleibt Future Work.
- Tests: `python -m unittest tests/test_cli_checkpoints.py -v` -> 6 Tests OK; `python -m unittest tests/test_pipeline_g1.py tests/test_cli_checkpoints.py -v` -> 12 Tests OK.
- Kein Render, keine Modellladung, keine Runtime-/Docker-/init.sh-/Backend-/n8n-/API-/GUI- oder Prompt-/Creative-System-Aenderungen.

## 2026-05-02 Phase G1 Pipeline Definitions, Checkpoints, Approval Gates
- Phase G1 umgesetzt: neuer declarativer Pipeline-Layer unter `agent_core/pipeline.py` und `agent_core/pipeline_defs/simple_video_v1.json`.
- `simple_video_v1` beschreibt den bestehenden Core-Flow mit `validate_job`, `create_plan`, `approve_plan`, `create_prompts`, `approve_prompts`, `generate_voice_optional`, `render_video`, `assemble` und `final_quality_gate`.
- Neue Schemas in `agent_core/schemas.py`: `PipelineDefinition`, `PipelineStepDefinition`, `PipelineRetryPolicy`, `PipelineApprovalPolicy`, `CheckpointRecord`, `CheckpointStatus`, `ApprovalMode`.
- `JobState` speichert jetzt `pipeline_id`, `checkpoints`, `current_checkpoint_id` und `blocked_by_checkpoint_id`; `StateStore` schreibt `checkpoints.json`.
- `VideoAgent.run_job()` setzt Checkpoints an den bestehenden Agent-Core-Phasen, ohne den Render-/Backend-Vertrag zu ersetzen.
- Approval-Gates sind lokal dateibasiert vorbereitet: bei `job.metadata.approval_gates_enabled=true` blockiert ein Gate wie `approve_plan` bis `/workspace/agent_runs/<job_id>/approvals/<checkpoint_id>.json` mit `approved=true` existiert.
- Fuer Tests und spaetere Operator-Checks gibt es `job.metadata.pipeline_dry_run=true`; der Lauf stoppt nach Plan-/Prompt-Checkpoints und startet keine Voice-/Storyboard-/Video-Backends.
- G1-Tests: `python -m unittest tests/test_pipeline_g1.py -v` -> 6 Tests OK.
- Zusaetzlich verifiziert: `python -m compileall -q agent_core tests/test_pipeline_g1.py` und `python -m unittest tests/test_planner_rules.py tests/test_creative_system.py -v` -> 19 Tests OK.
- Kein Render, keine Modellladung, keine Runtime-/Docker-/init.sh-/Backend-/n8n-/GUI-Aenderungen.

## 2026-05-01 Final Day Closeout: F2 Creative OS, Prompt Trace, CLI Live
- F2 Creative Operating System Grundlage umgesetzt: Hook Patterns, Shot Recipes und Anti-Patterns liegen als Playbook-Libraries unter `agent_core/creative_system/libraries/`.
- `morning_reset` Mode erweitert um Creative Goal, Audience Feel, Pacing, Hook Patterns, Shot Roles, Shot Recipe Order, Anti-Patterns, Quality Targets und Backend Prompt Policy.
- `clean_lifestyle_morning` Style erweitert um Preferred Camera Moves, Texture Targets, Object Count Max, Human Visibility Rules und positive Prompt Style Rules.
- Backend Prompt Policy eingefuehrt: Z-Image bekommt standardmaessig `positive_model_prompt` only; LTX bekommt `positive_model_prompt + kurze Avoid-Liste`.
- Pro Run wird neben `prompt_audit.json` jetzt `model_prompts.json` geschrieben: positive/negative/combined Prompts, `zimage_prompt_sent`, `ltx_prompt_sent`, Backend-Quellen, Shot Recipe, Hook Function und Leak Checks.
- `scripts/agent_core_cli.py` hat einen echten TTY-Live-Redraw-Modus mit `--live`/`--no-live`, Dashboard-Bloecken fuer System, Pipeline, Current Work, Prompt Preview, Szenen und Artefakte; Non-TTY bleibt Append-Log.
- Dry-Run `/workspace/agent_runs/phase-f2-creative-os-dry-run` erstellt: `prompt_audit.json` und `model_prompts.json` vorhanden, alle neuen Checks gruen, Z-Image positive-only, LTX short-avoid.
- Tests: `python3 -m unittest tests/test_creative_system.py tests/test_cli_live_dashboard.py tests/test_planner_rules.py tests/test_scene_planner.py tests/test_storyboard_pipeline.py tests/test_take_visual_review.py tests/test_output_quality_utils.py tests/test_final_quality_verdict.py` lief mit 70 Tests OK.
- Kein echter Render, keine Runtime-/Model-/init.sh-/Backend-/Director-Runtime-Aenderungen.

## 2026-05-01 Phase F1.1 Model Prompt Compiler Cleanup
- Phase F1.1 umgesetzt: Model-facing Prompts sind jetzt in `positive_model_prompt`, `negative_model_prompt` und kombinierten kurzen `model_prompt` getrennt.
- Grund: Phase F1 war formal audit-gruen, aber Model-Prompts waren noch zu lang, zu negativ, wiederholten Forbidden-Woerter und enthielten riskante Begriffe im positiven Prompt.
- `positive_model_prompt` ist jetzt kurze englische visuelle Prosa mit ca. 39-43 Woertern fuer Morning Reset; keine Debuglabels, keine deutschen Script-Snippets, keine langen No-Listen.
- `negative_model_prompt` ist eine separate kurze Begriffsliste mit maximal 25 Terms; der kombinierte `model_prompt` nutzt nur `positive + Avoid: kurze Liste`.
- Positive und negative Regeln sind getrennt: `single full-frame shot`, `one continuous scene`, `one clear water glass only` und `plain empty wooden table` werden nicht mehr als negative `no ...` Regeln behandelt.
- Bugfix: `no single full-frame shot` und `no one continuous scene` koennen in Morning-Reset-Model-Prompts nicht mehr entstehen.
- Positive risky words wie readable, text-bearing, phone, screen, ui, app, website, typography, letters, numbers, label/logo werden aus `positive_model_prompt` entfernt; `readable faces/depth` wird zu sicheren Formulierungen.
- Prompt Audit erweitert um Wortzaehlung, positive-risky-term Check, getrennten Negative-Prompt-Check, positive-constraints-in-negative Check, Overlong-Check und Repetition-Check.
- Dry-Run `/workspace/agent_runs/phase-f1-1-morning-reset-prompt-clean-dry-run`: alle neuen Audit-Checks true, Model-Prompts <= 140 Woerter, Positive-Prompts <= 100 Woerter.
- Kein Render, keine Runtime-/Model-/CLI-/init.sh-/Backend-Aenderungen.

## 2026-05-01 Phase F1 Creative Operating System + Prompt Audit
- Phase F1 gestartet und umgesetzt: Creative Operating System mit Mode-/Style-/Prompt-/Library-Struktur unter `agent_core/creative_system/`.
- Grund: Die Morning-Reset-Mini-Fixes reichten nicht; `quality-morning-reset-008` zeigte weiter sichtbaren Text aus model-facing Debug-/Strukturprompt (`WORLD / SETTING ...`) im Bild.
- Neuer `morning_reset` Mode definiert drei feste Playbook-Motive: `curtain_opening_window_light`, `water_glass_empty_table`, `calm_breathing_open_window`.
- Neuer Style `clean_lifestyle_morning` definiert weiche Morning-Light-Optik, wenige physische Objekte, keine Devices, keine UI-Layouts, keine Graphic-Design-Elemente.
- Prompt-Schicht trennt jetzt `debug_prompt` von `model_prompt`: Debug darf Sektionen enthalten, model-facing Prompt fuer Z-Image/LTX ist kurze englische visuelle Prosa ohne Debuglabels oder Script-Snippets.
- Z-Image Storyboard Adapter bevorzugt `effective_model_prompt`/`model_prompt`; Fallbacks werden kompiliert, damit keine Debug-Sektionslabels an Z-Image gehen.
- LTX2 Adapter bevorzugt den Take-`model_prompt` aus Step-Params statt globalem Debug-/Plan-Prompt.
- Prompt Audit wird pro geplantem Run als `prompt_audit.json` gespeichert mit mode/style, Model-Prompts, Motiven, Forbidden Visuals, Leaktermen und Checks.
- Qwen3-VL Reviewer nutzt jetzt den Creative-System-Reviewer-Systemprompt als JSON-only/visible-content Review-Anweisung.
- Dry-Run `/workspace/agent_runs/phase-f1-morning-reset-dry-run`: `mode_id=morning_reset`, alle Motive aus Playbook, keine `WORLD / SETTING`-/Script-Leaks in model prompts, Audit-Checks gruen.
- Keine Runtime-, Modell-, CLI-, init.sh-, Director-Runtime- oder Medienbackend-Aenderungen.

## 2026-05-01 Quality Morning Reset 008 Narration Isolation Fix
- Qualitaetsfix nach echtem `quality-morning-reset-007` umgesetzt; keine CLI-, init.sh-, Modell-, Runtime-, Director-, LTX/Z-Image/TTS/ACE- oder grosse Refactor-Aenderungen.
- Diagnose `quality-morning-reset-007`: `Vorhang auf`, `Stell ein Glas Wasser ab` und `Atme ruhig am Fenster` leakten als `environment`/`STORY BEAT` in visuelle Scene-/Take-Prompts; Qwen3-VL sah in Scene 1 sichtbaren Text/UI im Vordergrund.
- Visual Prompt Isolation: Script/Narration bleibt fuer Voice/Timing/Intent, waehrend visuelle Prompts englische Handlungsbeschreibungen verwenden, z. B. plain fabric curtains, one clear water glass only, calm breathing by an open window.
- PromptBuilder schreibt fuer Bild-/Video-Prompts keine deutschen Imperative oder title-artigen `Morning Reset:`-Snippets mehr in positive Motivfelder.
- Take-Review-Metadaten werden vor Selection zentral normalisiert; `passed` + Score 0.0 wird auf mindestens 0.7 korrigiert, Parser-/Missing-Score-Faelle werden `needs_review`.
- Selection liest den finalen `take_visual_review`-Payload bevorzugt vor stale Top-Level-Feldern, damit ein spaeter aktualisierter Qwen3-VL-Reject vor finaler Auswahl wirkt.
- Qwen3-VL Device/UI-Risiko ist fuer Social/Morning-Reset strenger: sichtbare UI/Device-Hits werden rejected statt als brauchbares `passed`/gutes `needs_review` weiterzureichen.
- Dry-Run `/workspace/agent_runs/quality-morning-reset-008-plan-dry-run`: keine deutschen Imperative in visual prompts/storyboard effective prompts; Forbidden Visuals enthalten text/ui/phone/split/collage; selected keyframes passed.

## 2026-05-01 Quality Morning Reset 007 Gate Fix
- Naechster echter Qualitaetsfix nach `quality-morning-reset-006` umgesetzt; keine CLI-, init.sh-, Modell-, Runtime-, Director-, LTX/Z-Image/TTS/ACE- oder Batch-/Wizard-Aenderungen.
- Diagnose `quality-morning-reset-006`: Scene 1 Fake-Text/Typografie, Scene 2 schwarzes Smartphone/Device neben Wasserglas, Scene 3 Split-Screen/Collage/Panel-Look mit eingebettetem Text; Qwen3-VL lieferte teils non-json.
- Take-Selektion geschaerft: technisch valide `passed` vor `needs_review` vor `rejected`; `rejected` darf nur noch als `selection_reason=last_resort_no_better_candidate` selected werden.
- Score/Status-Konsistenz: `passed` wird auf mindestens `postability_score=0.7` normalisiert; Qwen3-VL Parser-Warnungen oder invalide Scores koennen nicht mehr `passed` ergeben.
- Keyframe Visual Gate erweitert gegen visible/fake text, typography, letters/numbers, subtitles, phone/smartphone/black rectangle, screen/UI/app/web/browser, split-screen/collage/panels, paper/document/notebook, logo/label/sign.
- Morning-Reset-Motive konkretisiert: Scene 1 plain fabric curtains + blank wall; Scene 2 one clear water glass only auf plain empty wooden table, no second object/no phone/no black rectangle; Scene 3 single full-frame shot/one continuous scene/no split screen/no panels/no collage/no embedded subtitles.
- Qwen3-VL Subprocess fordert nun STRICT JSON ONLY, extrahiert JSON aus Text, macht einen JSON-only Retry und faellt bei weiterem Parserfehler auf `needs_review` mit `qwen3_vl_parser_warning` zurueck.
- Plan-/Storyboard-Dry-Run: `/workspace/agent_runs/quality-morning-reset-007-plan-dry-run`; alle selected keyframes `contract_preserved=true`, `visual_risk_status=passed`, positive Prompt-No-Gos fuer Scene 2/3 sauber.

## 2026-04-30 Day-End Backup / Handoff
- Tagesabschluss vorbereitet: CLI E2/E2.1/E2.2, Qwen3-VL-Isolation und erster Morning-Reset-Quality-Fix sind dokumentiert und werden schlank archiviert.
- `quality-morning-reset-006` ist der aktuelle technische Beleg nach dem ersten Quality-Fix: `success=True`, `final_phase=assembled`, Qwen3-VL real aktiv, aber Final Quality `failed`.
- Offene Diagnose fuer morgen: Scene 1 Fake-Text, Scene 2 Phone/Device-Risiko neben Wasserglas, Scene 3 Split-Screen/Collage/Text/UI-Drift, rejected selected Take Bug, Qwen3-VL non-json/parser warning.
- Naechster Schritt bleibt bewusst ungefixt fuer morgen: rejected Take Selection verhindern, Hard Keyframe Visual Gate gegen Text/Phone/Split-Screen, Qwen3-VL JSON-Robustheit, Morning-Reset-Motive weiter konkretisieren.
- Backup-Regel bleibt: keine Modelle, Venvs, Caches, GGUF/Safetensors oder grosse Runtime-/Checkpoint-Ordner archivieren; Qwen3-VL-Venv wird per Ensure-Script reproduziert.

## 2026-04-30 First Morning Reset Output Quality Fix
- Erster echter Output-Qualitaetsfix nach Phase E2 umgesetzt; kein CLI-, Runtime-, Dependency-, Modell-, Director-, Init- oder Backend-Umbau.
- Ursache aus `quality-morning-reset-005`: Social-/Content-/UI-/Phone-Begriffe konnten als visuelle Motive in Morning-Reset-Prompts rutschen; Scene 2 zeigte ein phone/screen-artiges Objekt, Scene 3 driftete Richtung UI/App/Web-Ausschnitt.
- `PromptBuilder` fuehrt jetzt einen Visual Prompt Sanitizer fuer positive visuelle Felder und `allowed_props`: Meta-/Formatbegriffe wie `social clip`, `reel`, `content`, `website`, `app`, `ui`, `screen`, `phone`, `browser`, `dashboard` werden aus Motivfeldern entfernt.
- Morning-Reset-Motivbibliothek geschaerft: Vorhang/Fensterlicht, Wasserglas auf Holzoberflaeche, Handbewegung, Pflanzen/Stoff/Licht und ruhiges Atmen am Fenster statt Smartphone-/Screen-/App-Motive.
- `allowed_props` werden von Device-/UI-/Text-/Paper-Begriffen bereinigt; `forbidden_props` enthalten diese Begriffe weiterhin hart inklusive phones, screens, user interface, app layout, website, social media frame, split screen und collage.
- PromptBuilder wiederholt Device-/UI-Verbote nahe am Motiv ueber `MOTIF SAFETY` und schuetzt Storyboard-`effective_prompt` gegen UI/mockup/collage/split-screen Drift.
- `readable human action` wurde aus positiven Prompts durch `clear human action` ersetzt.
- Take-/Keyframe-Review erkennt positive Phone/Screen/UI/App/Website-Hinweise strenger; Qwen3-VL-Hinweise auf sichtbare Device-/UI-Risiken koennen nicht mehr als `passed` stehen bleiben.
- Plan-only Dry-Run unter `/workspace/agent_runs/quality-morning-reset-006-plan-dry-run` erzeugt `plan.json`, `scene_plan.json` und `storyboard_plan.json` ohne Medienrender; positive Prompt-Verletzungen: keine.

## 2026-04-30 Phase E2.2 CLI Dashboard Polishing
- Phase E2.2 als reines CLI-Dashboard-Polishing umgesetzt; keine Pipeline-, Quality-, Prompt-, Modell-, Director-, Backend- oder Init-Aenderungen.
- Duplicate-Video-Label im Pipeline-Block gefixt: `Video Backend`, `Vision Review`, `Render` und `Assembly` sind jetzt getrennt.
- Vision-Review-Status ist klarer: `qwen3_vl · real inference used`, `qwen3_vl · parser warning`, `qwen3_vl · runtime missing`, `qwen3_vl · no real inference` oder `heuristic`.
- Quality-Ausgabe gruppiert Meldungen jetzt in `QUALITY ISSUES`, `VISION RUNTIME WARNINGS`, `VISION REVIEW WARNINGS` und `POLICY / CONFIG WARNINGS`.
- Subtitle-Burn-Konflikt wird explizit erklaert: `subtitle-mode=burn` fuegt sichtbaren Text hinzu und ist fuer clean no-text visual tests nicht geeignet.
- Scene Summary zeigt Status, Take, Score mit zwei Nachkommastellen, Provider und kurze Warning-Marker.
- Next Actions sind regelbasiert smarter: Hinweise fuer Subtitle-Off/Sidecar, Qwen3-VL-JSON-Warnungen, Final-Frame-Rejects und needs-review-Takes.

## 2026-04-30 Phase E2 CLI Dashboard / Produktionsansicht
- Phase E2 als reiner CLI-Output-/Darstellungs-Task umgesetzt; keine Pipeline-, Prompt-, Quality-, Modell-, Director-, Backend- oder Init-Aenderungen.
- `scripts/agent_core_cli.py` rendert jetzt einen dashboard-artigen Run-Header mit Job, Format, Mode, Prompt und Startzeit sowie einen klaren System-/Mode-Block.
- Live-Polling ist kompakter: Ausgabe nur bei Status-/Phasen-/Detailaenderung oder periodischem Heartbeat; keine erfundenen Prozentwerte, stattdessen echte Scene-/Take-/Elapsed-Informationen soweit verfuegbar.
- Neue Dashboard-Bloecke fuer Progress, Current Step, Scene Summary, Quality Live, Success Summary und Failure Summary.
- `--inspect-run` nutzt dieselben Success-/Failure-Dashboards fuer vorhandene lokale Runs.
- Failure-Ausgabe extrahiert Root-Cause-Zeilen aus Backend-`job.log`-Tails und zeigt bekannte Bedeutungen fuer `tokenizer.model`, `SiglipVisionModel.vision_model`, CUDA-OOM, Qwen3-VL-Runtime und Importfehler.
- Next Actions enthalten Inspect-Command, Video-Pfad bzw. Next Debug Command fuer Backend-Logs.

## 2026-04-30 Final Backup / Handoff
- Abschlussstand fuer Download/Restore gesichert: schlankes Projektarchiv ohne Modelle, Venvs, Caches, GGUF oder Safetensors wird erstellt.
- Reproduzierbarkeit fuer Qwen3-VL-Review-Venv ergaenzt: `/workspace/scripts/ensure_qwen3_vl_review_runtime.sh` erstellt `/workspace/venvs/qwen3-vl-review` mit System-Site-Packages und gepinnten Review-Dependencies.
- Restore-Hinweise in `/workspace/codex/HANDOFF.md`: `init.sh`, Qwen3-VL-Venv-Ensure, FastAPI/Director-Checks und erster Inspect-Run.
- Heutiger Endstand bleibt: init stabilisiert, Gemma/LTX readiness gefixt, Phase E CLI Produktions-Cockpit umgesetzt, CLI Vision Flags und Provider-Wiring aktiv, Qwen3-VL per Venv/Subprocess isoliert.
- `quality-morning-reset-003` bleibt der letzte echte Beleg: technisch erfolgreich assembled, Final Quality `needs_review` wegen echter sichtbarer Text-/Papier-/Subtitle-Risiken.

## 2026-04-30 Qwen3-VL Dependency Isolation Fix
- Das globale Qwen3-VL-Transformers-Upgrade auf `5.7.0` brach den LTX/Gemma-Pfad: `quality-morning-reset-002` scheiterte in LTX mit `AttributeError: 'SiglipVisionModel' object has no attribute 'vision_model'`.
- Main-/FastAPI-Runtime wurde wieder LTX-kompatibel gesetzt: globales `transformers==4.52.4`; globales `kernels` wurde entfernt, weil es mit dem alten `huggingface_hub` den LTX-Import blockierte.
- Qwen3-VL laeuft jetzt isoliert in `/workspace/venvs/qwen3-vl-review` mit System-Site-Packages, eigener `transformers 5.7.0`-Installation und `kernels 0.13.0`; Torch wird aus der bestehenden Systeminstallation wiederverwendet.
- Neues Subprocess-Script `/workspace/scripts/qwen3_vl_review_subprocess.py` nimmt JSON auf stdin und gibt JSON auf stdout zurueck.
- `agent_core/utils.py` ruft fuer `provider=qwen3_vl` standardmaessig `/workspace/venvs/qwen3-vl-review/bin/python /workspace/scripts/qwen3_vl_review_subprocess.py` auf; konfigurierbar via `QWEN3_VL_PYTHON`, `QWEN3_VL_REVIEW_SCRIPT` und `QWEN3_VL_REVIEW_TIMEOUT_SEC`.
- Isolierter Qwen3-VL-Smoke ueber `evaluate_take_visual_review()` ist gruen: `real_vlm_inference_used=True`, `provider=qwen3_vl`, Status `passed`.
- Echter Kontrollrun `quality-morning-reset-003` war technisch erfolgreich: Director, Voice, Storyboard, LTX und Qwen3-VL-Review liefen; `final.mp4` wurde assembled. Final Quality blieb `needs_review`, weil Qwen3-VL echte sichtbare Subtitle-/Text-/Papier-Risiken im finalen Video meldete.

## 2026-04-30 Qwen3-VL Runtime Fix
- Qwen3-VL Provider-Wiring war korrekt: CLI-Vision-Flags kommen im Job an und `visual_review_provider=qwen3_vl` wird gesetzt.
- Echter Fehler war die FastAPI/Worker-Runtime: `/usr/bin/python` nutzte `transformers 4.52.4`, das `model_type=qwen3_vl` nicht kannte.
- Root-/FastAPI-Python unter `/usr/bin/python` bzw. `/usr/bin/python3.12` wurde in diesem Zwischenschritt gezielt aktualisiert; dieser Ansatz wurde danach durch den Qwen3-VL-Dependency-Isolation-Fix ersetzt, weil er LTX/Gemma brach.
- Keine Torch-/CUDA-Reinstallation, keine Modell-Downloads und kein Init-/Backend-/Prompt-Umbau.
- Qwen3-VL Runtime-Smoke ueber `evaluate_take_visual_review()` ist gruen: `provider=qwen3_vl`, `real_vlm_inference_used=True`, Status `passed`, keine Qwen3-VL-Warnungen, Laufzeit ca. `57.04s`.
- FastAPI wurde nach dem Dependency-Fix neu gestartet; `/health` und Director `/v1/models` antworten.

## 2026-04-30 Vision Review Provider Wiring
- CLI-to-Agent Wiring fuer Vision Review umgesetzt, ohne Init, Modelle, Director, Backends oder grosse Phase-A-D-Logik umzubauen.
- Ursache: `VISION_REVIEW_PROVIDER=qwen3_vl` als Env vor dem CLI-Aufruf erreicht den bereits laufenden FastAPI/Worker-Prozess nicht; der Job lief deshalb weiter mit `heuristic`.
- Neue CLI-Flags in `scripts/agent_core_cli.py`:
  - `--vision-review-enabled`
  - `--no-vision-review`
  - `--vision-review-provider {heuristic,qwen3_vl,none}`
  - `--vision-review-model-dir PATH`
  - `--vision-review-max-frames N`
- Die CLI schreibt Vision-Settings in `job.metadata`; `ProductionPlanner` uebernimmt sie nach `plan.metadata`.
- `VideoAgent` und `ResultAssembler` geben `vision_review_enabled`, Provider, Modellordner und Max-Frames aus `plan.metadata` an Take Visual Review und Final Quality Verdict weiter.
- Provider-Aufloesung bevorzugt explizite Job-/Plan-Metadata vor Env; ein Job mit `vision_review_enabled=true` und `vision_review_provider=qwen3_vl` kann damit eine im Serverprozess anders gesetzte Env uebersteuern.
- CLI-Startausgabe und Quality Summary zeigen Vision-Review-Settings, `final_frame_review.provider` und ob realer VLM-Pfad genutzt wurde.

## 2026-04-30 Phase E CLI Produktions-Cockpit
- Phase E gestartet und fuer die bestehende CLI umgesetzt, ohne Pipeline-, Backend-, API-, Modell- oder Init-Umbau.
- `scripts/agent_core_cli.py` zeigt beim Jobstart jetzt Job-ID, Idee/Script-Kurzform, Dauer/Aufloesung/Orientierung, Voice/Storyboard/Music/Subtitles, Takes/Variations und API-Endpunkt.
- Polling-Ausgabe ist strukturierter: Status-/Phasenwechsel mit elapsed time, Phase elapsed, Summary, Director-Status, Step-Status und Take-/Scene-Zusammenfassung; Heartbeat nur periodisch statt stumpf identischer Ausgabe.
- Director-Zusammenfassung zeigt `director_mode`, `director_llm_active`, Provider, Modell und Fallback-Grund, sobald aus lokalen Artefakten sichtbar.
- Abschlussausgabe zeigt Success Summary mit `final.mp4`, Datei-Groesse, Dauerwerten, Result-/State-/Takes-Pfaden und Final Quality Verdict.
- Fehlerausgabe zeigt jetzt Error Summary mit Phase, Scene, Take, Backend, Backend-Job-ID, Agent-Fehler, Backend-Fehler, Artefaktpfaden und optionalem Backend-`job.log`-Tail.
- Neue CLI-Optionen: `--tail-error-log-lines`, `--no-log-tail`, `--quiet`, `--verbose`, `--inspect-run`.
- Offline-Fixtures geprueft: `readiness-small-social-003` als Success Summary und `readiness-small-social-001` als LTX-Failure mit Backend-Logtail.

## 2026-04-30 LTX/Gemma Readiness Fix
- Echten Fehler aus `readiness-small-social-002` analysiert: LTX scheiterte nicht mehr an `tokenizer.model`, sondern an fehlendem `preprocessor_config.json` unter `/workspace/LTX-2/checkpoints/gemma-3`.
- Gemma-Ordner war unvollstaendig: `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `model.safetensors.index.json` und Shards `model-00003-of-00005.safetensors` bis `model-00005-of-00005.safetensors` fehlten bzw. waren nur als unvollstaendige Cache-Spuren vorhanden.
- Nur fehlende Gemma-Dateien aus `google/gemma-3-12b-it-qat-q4_0-unquantized` mit `HF_HUB_ENABLE_HF_TRANSFER=0` und `HF_HUB_DISABLE_XET=1` nachgeladen; keine Modelle geloescht.
- Gemma-Indexcheck danach gruen: 5/5 Shards vorhanden, keine `.incomplete` und keine 0-byte Dateien.
- `init.sh` minimal gehaertet: Gemma gilt nicht mehr nur wegen `config.json` als fertig, sondern erst mit `tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `model.safetensors.index.json` und allen Index-Shards.
- Nicht-rendernder LTX/Gemma-Smoke gruen: `module_ops_from_gemma_root()` akzeptiert den Gemma-Root, `TI2VidTwoStagesPipeline` importiert.
- `readiness-small-social-003` lief erfolgreich mit Director `llm_augmented`, Qwen-TTS, LTX-Video und muxed `final.mp4` unter `/workspace/agent_runs/readiness-small-social-003/final.mp4`.

## 2026-04-30 Final Init Stabilisierung
- `/workspace/init.sh` bleibt klein und OG-basiert; es wurde kein Guard-/Heartbeat-/Stall-Framework eingebaut.
- Minimaler `flock`-Lock ueber `/workspace/status/init.lock` verhindert parallele Init-Laeufe; ein zweiter Lauf beendet sich mit klarer Meldung.
- `hf_transfer` ist nicht mehr Default, weil es auf RunPod bei grossen Downloads mit `no permits available`, `.incomplete`-/Lock-Haengern und futex-wartenden Python-Prozessen instabil war.
- Stabiler Default ist jetzt der normale HuggingFace-Downloader: `HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"`; Speed bleibt optional per `HF_HUB_ENABLE_HF_TRANSFER=1 bash /workspace/init.sh`.
- Xet bleibt standardmaessig aus: `HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"`.
- Die `hf_download_*`-Helper loggen Repo/Datei/Ziel und aktiven hf_transfer-Status, skippen vorhandene Dateien bzw. vorhandene Snapshot-Zielordner und setzen Ready-Flags nur nach erfolgreichem Helper-/Verify-Lauf.
- Qwen3-VL bleibt optional ueber `Qwen3_VL_Review=on` oder `Vision_Review_Model=on`; Verify/Download laeuft weiterhin ueber `/workspace/scripts/download_qwen3_vl_model.py`.
- Keine Aenderungen an `agent_core`, App/API, Tests, Modellen, llama.cpp oder Phase E.

## 2026-04-30 Init Restore auf kleine OG-Basis
- `/workspace/init.sh` wurde wieder aus der kleinen, funktionierenden `init(OG).sh` aufgebaut; die grosse Guard-/Heartbeat-/Lock-Version aus `init(fehler).sh ` wurde bewusst nicht weitergefuehrt.
- Aus der grossen Version wurden nur minimale Fixes uebernommen:
  - Non-Interactive-Env fuer apt/git/HF/pip inklusive HF-Transfer-/Xet-Defaults
  - Director-Autostart prueft jetzt auf vorhandene `serve_director_llm.sh`, setzt best-effort `chmod +x` und startet per `DIRECTOR_LLM_DAEMON=1 bash ...`
- Qwen3-VL wurde nicht als grosser Shell-Block in `init.sh` eingebaut, sondern ueber `/workspace/scripts/download_qwen3_vl_model.py`.
- Qwen3-VL-Download ist optional; `tools.config` enthaelt den expliziten Schalter `Qwen3_VL_Review`, Aktivierung ueber `Qwen3_VL_Review=on` oder `Vision_Review_Model=on`.
- Keine Aenderungen an `agent_core`, API, App, llama.cpp, Runtime-Dateien oder Modellbestaenden.

## 2026-04-29 Phase D Final Quality Verdict
- Qwen3-VL echter Bild-Smoke vor Phase D erfolgreich:
  - Testbild: `/workspace/status/qwen3_vl_smoke/clean_test_image.jpg`
  - Ergebnis: `provider=qwen3_vl`, `take_visual_review_status=passed`, `postability_score=1.0`
  - Laufzeit des zweiten sauberen Smokes: ca. `10.983s`
  - Ergebnis-JSON: `/workspace/status/qwen3_vl_smoke/qwen3_vl_smoke_result.json`
- Phase D umgesetzt, ohne Phase E, API-, GUI-, Init-/Startup-, Runtime-, Director-/Qwen3.6- oder Medienbackend-Umbau.
- `agent_core/utils.py` fuehrt jetzt `evaluate_final_quality_verdict()` ein.
- Final Quality Verdict kombiniert:
  - technische `final.mp4`-Validation
  - Assembly-Metadata
  - `selected_scene_outputs`
  - Phase-C-`take_visual_review`
  - Phase-B2-Keyframe-`visual_risk_review`, falls vorhanden
  - Subtitle-/Overlay-Metadata
  - Voice-/Music-Metadata
  - wenige extrahierte Final-Frames, optional mit Qwen3-VL, sonst heuristisch/metadata-basiert
- `ResultAssembler` schreibt `metadata.final_quality_verdict` und spiegelt den Verdict in `metadata.assembly.final_quality_verdict` sowie in die Final-MP4-Artefakt-Metadata.
- Failure-Resultate erhalten ebenfalls einen expliziten `final_quality_verdict` mit `final_quality_status=failed`.
- Neue Verdict-Felder:
  - `final_quality_status`
  - `final_postability_score`
  - `main_issues`
  - `warnings`
  - `problem_scenes`
  - `recommended_next_action`
  - `quality_policy_version`
  - `quality_sources`
- Neuer Test: `tests/test_final_quality_verdict.py` fuer passed/needs_review/failed/Metadata und GPU-freie Provider-/Frame-Review-Abdeckung.
- Kein vorhandener kleiner `agent_runs/**/final.mp4` lag fuer einen zusaetzlichen Light-Smoke vor; es wurde bewusst kein neuer GPU-Render gestartet.

## 2026-04-29 Phase C Take Visual Review / Postability Score
- Phase C umgesetzt, ohne Init-/Startup-, Runtime-, Director-, Backend-, API-, GUI- oder Phase-D/E-Umbau.
- `agent_core/utils.py` fuehrt jetzt `extract_review_frames()` ein:
  - nutzt `ffprobe` fuer Duration
  - extrahiert per `ffmpeg` 1 bis 5 Review-JPGs pro technisch validem MP4-Take
  - speichert pro Frame `timestamp_sec`, `path`, `exists` und `file_size_bytes`
  - fehlt `ffmpeg`/`ffprobe` oder ist das Video defekt, wird gewarnt statt der Core hart gecrasht
- `evaluate_take_visual_review()` bewertet pro Take:
  - technische Video-Validation
  - Scene World Contract
  - positive riskante Inhalte in Subject/Action/Allowed Props
  - positive Prompt-Risiken ausserhalb von Forbidden-/No-/Text-Risk-Policy-Clauses
  - optional vorhandenen `visual_risk_review` des selektierten Keyframes
  - Review-Frame-Extraktion
- Neue Take-Metadata:
  - `take_visual_review`
  - `take_visual_review_status`
  - `postability_score`
  - `visual_review_provider`
  - `review_frames`
  - `scene_contract_summary`
- Take-Auswahl priorisiert jetzt technisch valide Takes nach `passed` vor `needs_review` vor `rejected`, danach nach hohem `postability_score`, technischem Score und kreativer Heuristik.
- Optionaler Provider `VISION_REVIEW_PROVIDER=qwen3_vl` ist lazy eingebaut und nutzt den lokalen Modellordner `/workspace/models/Qwen3-VL-4B-Instruct-FP8`; Default bleibt heuristisch, damit Unit Tests und normale Runs ohne GPU-/VLM-Zwang laufen.
- Qwen3-VL-Inferenz wurde nicht als Pflicht-Smoke gefahren; bei fehlenden Frames, fehlendem Modell, Dependency- oder Inferenzfehlern bleibt das Ergebnis ehrlich `needs_review` bzw. bereits technische `rejected`-Bewertungen werden nicht weichgespült.
- Neue Tests: `tests/test_take_visual_review.py` fuer Frame Extraction, Heuristik, False-Positive-Schutz, Auswahlprioritaet und Persistenz der Metadata.
- Verifikation:
  - `python3 -m unittest tests/test_output_quality_utils.py`
  - `python3 -m unittest tests/test_storyboard_pipeline.py`
  - `python3 -m unittest tests/test_scene_planner.py`
  - `python3 -m unittest tests/test_planner_rules.py`
  - `python3 -m unittest tests/test_assembler_mux.py`
  - `python3 -m unittest tests/test_take_visual_review.py`

## 2026-04-29 Qwen3-VL Model Setup Verify
- Nur Qwen3-VL-Modell-Setup und Verify umgesetzt; keine Phase C, kein VisionReviewAdapter, kein agent_core-, Pipeline-, Init-/Startup- oder Director-Umbau.
- Gewaehltes Vision-Review-Modell fuer spaetere Phase C/D lokal abgelegt:
  - Repo: `Qwen/Qwen3-VL-4B-Instruct-FP8`
  - Zielordner: `/workspace/models/Qwen3-VL-4B-Instruct-FP8`
- Download lief per `huggingface_hub.snapshot_download` direkt in den Zielordner, mit `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_HUB_DISABLE_XET=1`, Non-Interactive-Env und Resume/Skip-Verhalten.
- Dateiverifikation gruen:
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `preprocessor_config.json`
  - `video_preprocessor_config.json`
  - `model.safetensors.index.json`
  - `model-00001-of-00002.safetensors`
  - `model-00002-of-00002.safetensors`
  - keine `.incomplete`-Dateien im Zielordner
- Modellordnergroesse: ca. `5.7G`; Shard-Groessen: `5366863440` und `654372016` Bytes.
- Root-Dependencies minimal fuer Load-Smoke angepasst: `transformers==4.57.3`, `tokenizers==0.22.2`, `qwen-vl-utils==0.0.14`.
- Load-Smoke gruen:
  - `AutoConfig.from_pretrained(...)` erkennt `Qwen3VLConfig`, `model_type=qwen3_vl`
  - `AutoProcessor.from_pretrained(...)` laedt `Qwen3VLProcessor`
  - `AutoModelForImageTextToText.from_pretrained(..., device_map="cpu")` laedt `Qwen3VLForConditionalGeneration`
- Keine Vision-Review-Logik gebaut; naechster Schritt bleibt Phase C Take Visual Review / Postability Score mit optionalem Qwen3-VL Provider.

## 2026-04-29 Init Download Freeze Hardening
- Nur Init-/Download-/Startup-Pfad bearbeitet; kein Phase-C-Bau, kein Qwen3-VL, kein Adapter- oder agent_core-Refactor.
- Reale Freeze-Ursache im laufenden Pod gefunden: ein alter `init.sh`-Prozess hing in `huggingface_hub`/Xet beim Z-Image-Turbo-Snapshot, hielt eine HF-Lockdatei auf einem `.incomplete`-Blob und schrieb seit Minuten keine Bytes mehr; ein zweiter Init-Lauf wartete dahinter und sah wie ein weiterer Freeze aus.
- `init.sh` setzt jetzt Non-Interactive-Guards fuer Git/HF/Pip (`GIT_TERMINAL_PROMPT=0`, `GCM_INTERACTIVE=never`, `HF_HUB_DISABLE_TELEMETRY=1`, `PYTHONUNBUFFERED=1`, `PIP_NO_INPUT=1`).
- `init.sh` nutzt jetzt ein `flock`-basiertes Init-Lock, damit parallele Init-/Download-Laeufe nicht mehr dieselben HF-Locks blockieren.
- HF-Downloads laufen jetzt ueber einen Guard mit Start-/Ende-/Fehler-Logging, Fortschritts-Heartbeat, Gesamt-Timeout, Stall-Timeout, Retry und Resume ueber `huggingface_hub`.
- Primaerpfad bleibt schnell: `hf_transfer` wird genutzt, wenn installiert; Xet ist standardmaessig deaktiviert, weil genau der Xet-Pfad im Pod eingefroren war.
- Lokale Snapshot-Skip-Pruefung ist jetzt gegen sharded `*.index.json` gehaertet; fehlende Shards wie `diffusion_pytorch_model-00002-of-00003.safetensors` zaehlen nicht mehr als fertig.
- `INIT_CHECK_ONLY=1` prueft Konfiguration/Pfade ohne Downloads oder Service-Starts; `INIT_SKIP_DOWNLOADS=1` markiert keine falschen `zimage_ready`-/`init_done`-Flags.
- Director-Modell-Download und Director-Autostart sind im Init-Pfad jetzt ebenfalls durch Guard bzw. Timeout begrenzt; der bestehende `serve_director_llm.sh`-Execute-Bit-Fix bleibt erhalten.
- `scripts/ensure_llama_cpp.sh` bekam nur Non-Interactive-Git/Pip-Env-Guards; keine llama.cpp-Runtime- oder Build-Logik wurde umgebaut.

## 2026-04-29 Director Init Completion
- Vorheriger Abschluss war unvollstaendig, weil der Director nach dem Skip-Smoke down war und das konfigurierte GGUF-Modell lokal fehlte.
- Erwarteter Director-Pfad aus `config/director_llm.env` bestaetigt:
  - Repo: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
  - Datei: `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - Ziel: `/workspace/models/director/qwen3.6-35b-a3b/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- GGUF ueber den vorgesehenen `scripts/download_director_model.py`-Pfad mit `HF_HUB_DISABLE_XET=1`, `HF_HUB_ENABLE_HF_TRANSFER=1`, Non-Interactive-Env und Timeout nachgeladen.
- Finale Datei liegt real am erwarteten Pfad mit `21391448384` Bytes.
- `scripts/ensure_llama_cpp.sh` fuehrte keinen Rebuild aus; vorhandene Runtime wurde nur repariert und `ldd` loest danach alle lokalen Libraries.
- Director wurde ueber den vorgesehenen `scripts/serve_director_llm.sh`-Pfad als Daemon gestartet.
- Verifikation danach gruen:
  - `curl http://127.0.0.1:8011/v1/models`
  - `python3 /workspace/scripts/check_director_llm.py`
  - laufender `llama-server` mit dem erwarteten GGUF
- `init.sh` hat zusaetzlich einen kleinen `INIT_DIRECTOR_ONLY=1`-Pfad fuer echte Director-Download-/Startup-Verifikation ohne `INIT_SKIP_DOWNLOADS`/`INIT_CHECK_ONLY`; normaler Init bleibt unveraendert.

## 2026-04-28 Phase B2 Keyframe Visual Risk Review
- Phase B2 umgesetzt, ohne Runtime-, Backend-, API-, GUI-, init/start- oder llama.cpp-Umbau.
- `agent_core/utils.py` fuehrt `evaluate_keyframe_visual_risk()` als leichten Contract-/Prompt-/Technik-Review fuer Storyboard-Keyframe-Kandidaten ein.
- Review-Metadata pro Kandidat:
  - `visual_risk_status`: `passed`, `needs_review` oder `rejected`
  - `risk_score`
  - `issues`
  - `warnings`
  - `policy_version`
  - `source`
  - `checked_contract_fields`
  - `checked_prompt_fields`
- False-Positive-Regel umgesetzt: Verbote in `forbidden_props`, `text_risk_policy` oder `no ...`-Promptteilen zaehlen nicht als positives Risiko; riskant sind positive Inhalte in Subject/Action/Allowed Props oder aktive Prompt-Anforderungen.
- Storyboard-Auswahl bevorzugt jetzt technisch valide Kandidaten in der Reihenfolge `passed` vor `needs_review` vor `rejected`.
- `storyboard_plan.json` und Kandidaten-Metadaten enthalten jetzt `visual_risk_review`.
- Dry-Run-Verifikation ohne GPU/Video-Render:
  - `/workspace/agent_runs/phase-b2-dry-morning-reset`
  - `/workspace/agent_runs/phase-b2-dry-focus-break`
- Keine finale Bildqualitaet behauptet; Phase C ist Take Visual Review / Postability Score, spaeter Phase D Final Quality Verdict und Phase E CLI Produktions-Cockpit.

## 2026-04-28 Phase B1 Storyboard Contract-Aware Prompts
- Phase B1 umgesetzt, ohne Runtime-, Backend-, API-, GUI- oder llama.cpp-Umbau.
- `ProductionPlanner.build_storyboard_render_plan()` baut pro Keyframe-Kandidat jetzt einen scene-specific `effective_prompt` aus Scene World Contract, Scene Prompt, Candidate Prompt und Variation-Kontext.
- `storyboard_step.params` speichert jetzt `effective_prompt`, `prompt_source`, `candidate_prompt_text`, `scene_prompt_text`, `scene_world_contract` und `storyboard_prompt_metadata`.
- `ZImageStoryboardAdapter` nutzt bevorzugt diesen `effective_prompt`; Fallback auf Candidate-/Global-Prompt bleibt erhalten.
- Storyboard-Reports koennen den effektiv genutzten Prompt und die Contract-Metadaten pro Kandidat nachvollziehen.
- Dry-Run-Verifikation ohne GPU/Video-Render:
  - `/workspace/agent_runs/phase-b1-dry-morning-reset`
  - `/workspace/agent_runs/phase-b1-dry-focus-break`
- Vision-/Keyframe-Eval wurde nicht gebaut; das bleibt Phase B2.

## 2026-04-28 Phase A Scene World Contract
- Phase A fuer den aktuellen Output-Quality-Fokus umgesetzt, ohne Backend-, Runtime-, API- oder llama.cpp-Umbau.
- `agent_core/prompt_builder.py` fuehrt jetzt einen kleinen Scene World Contract pro Szene ein und speichert ihn in `prompt_build_metadata.scene_world_contract`.
- Scene-Prompts werden jetzt als PromptBuilder v2 mit klaren Sektionen gebaut:
  - `WORLD / SETTING`
  - `SUBJECT / ACTION`
  - `CAMERA / LIGHTING`
  - `STYLE LOCK`
  - `ALLOWED VISUALS`
  - `FORBIDDEN VISUALS`
  - `TEXT RISK POLICY`
  - `SOCIAL FORMAT CONTRACT` bei aktivem Social-Tip-Guard
- Variation-Prompts behalten den World Contract aktiv und wiederholen die Forbidden-Visuals, damit Close-up-/Detail-Varianten keine Papier-, Screen- oder Textobjekte zurueckbringen.
- Social-Tip-Prompts sind haerter gegen lesbaren Text, Handschrift, Papier, Screens/UI, Labels, Logos, Poster, Signs, generierte In-Scene-Untertitel, Typografie/Glyphen/Buchstaben/Zahlen und Focus-Break-Desk-Drift.
- `agent_core/planner.py` wurde nur klein angepasst: die generische `tactile_detail`-Variation fordert jetzt clean surfaces statt interfaces.
- Tests:
  - `python3 -m unittest tests/test_planner_rules.py` gruen
  - `python3 -m unittest tests/test_assembler_mux.py` gruen
  - `python3 -m unittest tests/test_output_quality_utils.py` gruen
- Kein neuer GPU-Render und kein visuelles Output-Eval in Phase A; Storyboard scene-specific prompts und Keyframe Visual Eval bleiben Phase B.

## 2026-04-21 Fresh Startup Recheck
- Vorher-Zustand real festgehalten:
  - `uvicorn app.main:app` lief auf `8000`
  - `127.0.0.1:8011` lieferte real `Connection refused`
  - `init.sh` und `scripts/ensure_llama_cpp.sh` waren lokal geaendert, aber der frische Startup-Pfad war noch nicht neu bewiesen
- kompletter Pod-Neustart war in der Session nicht praktikabel; deshalb den engsten realistischen frischen Startpfad direkt ueber `bash /workspace/init.sh` gefahren
- wichtiger Befund:
  - `init.sh` hat den Director ohne manuelles Vorstarten von `scripts/serve_director_llm.sh` selbst hochgebracht
  - danach liefen `uvicorn app.main:app` und `llama-server` parallel sauber
  - `curl http://127.0.0.1:8011/v1/models` und `python3 /workspace/scripts/check_director_llm.py` waren danach gruen
- kleiner echter Produktivcheck direkt danach:
  - Job `startup-recheck-20260421` ueber den produktiven API-/CLI-Pfad gestartet
  - `director_mode=llm_augmented`
  - `director_llm_active=true`
  - `director_fallback_reason=null`
  - `final.mp4` erfolgreich unter `/workspace/agent_runs/startup-recheck-20260421/final.mp4`
- Einordnung:
  - der `init.sh`-Autostart-Fix ist jetzt nicht mehr nur indirekt oder ueber manuellen Director-Start belegt
  - ein kompletter Pod-Neustart bleibt zwar weiter ein eigener noch strengerer Beleg, war in diesem Lauf aber bewusst nicht der durchgefuehrte Pfad

## 2026-04-20 Director Restore Runtime Debug
- konkreten Fallback-Fall `cli-test-basic-001` forensisch geprueft
- belastbar bestaetigt:
  - kein Payload-Fehler
  - kein Config-Fehler
  - kein fehlender `llama.cpp`-Build
  - echter Fallback-Grund war `director_llm_request_failed: <urlopen error [Errno 111] Connection refused>`
- direkte Ursache im Startup-Pfad belegt:
  - `init_download.log` zeigte `WARN: /workspace/scripts/serve_director_llm.sh missing or not executable; skipping auto-start`
  - `init.sh` pruefte den Director-Autostart ueber `-x`, bevor spaeter im selben Skript erst `chmod +x /workspace/scripts/*.sh` lief
  - dadurch blieb der lokale Director-Serve trotz vorhandenem Modell und vorhandener Runtime beim Pod-Start unten
- minimaler Fix:
  - `init.sh` setzt fuer den Director-Autostart `serve_director_llm.sh` jetzt vor dem Start explizit auf executable und ruft es per `bash` auf
- danach real verifiziert:
  - `DIRECTOR_LLM_DAEMON=1 /workspace/scripts/serve_director_llm.sh`
  - `curl -fsS http://127.0.0.1:8011/v1/models`
  - `python3 /workspace/scripts/check_director_llm.py`
  - ein echter kleiner CLI-Live-Run `cli-test-basic-001-reverify` lief wieder mit `director_mode=llm_augmented`
- kein `llama.cpp`-Rebuild noetig

## 2026-04-20 llama.cpp Runtime Verification
- vorhandenen `llama.cpp`-Runtime-/Build-Stand im aktuellen Pod ohne Rebuild erneut verifiziert
- bestaetigt: unter `/workspace/tools/llama.cpp/build/bin` lagen reale ELF-Artefakte fuer `llama-server`, `llama-cli` und die versionierten `libggml*`, `libllama*` und `libmtmd`-Libraries bereits vor
- realer Sonderfall im aktuellen Snapshot:
  - `llama-server` und `llama-cli` hatten nur Modus `644`
  - die echten versionierten `.so.*`-Dateien waren da, aber die Linux-Loader-Aliase `.so.0` und `.so` fehlten
  - dadurch meldete `ldd` zunaechst `not found`, und `scripts/ensure_llama_cpp.sh` haette faelschlich einen Rebuild angestossen
- minimaler Fix ausschliesslich in `tools/llama.cpp/build/bin` umgesetzt:
  - Execute-Bit fuer `llama-server` und `llama-cli` gesetzt
  - Symlink-Ketten fuer `libggml-base`, `libggml-cpu`, `libggml-cuda`, `libggml`, `libllama-common`, `libllama` und `libmtmd` angelegt
- danach erneut verifiziert:
  - `ldd /workspace/tools/llama.cpp/build/bin/llama-server` loest alle lokalen Libraries korrekt auf
  - `llama-server --help` und `llama-cli --help` laufen erfolgreich
  - `scripts/ensure_llama_cpp.sh` meldet jetzt korrekt `llama.cpp already available`
  - kurze echte Serve-Probe ueber `DIRECTOR_LLM_DAEMON=1 /workspace/scripts/serve_director_llm.sh` erfolgreich
  - `/v1/models` und `python3 /workspace/scripts/check_director_llm.py` antworteten erfolgreich mit dem realen Modell `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- wichtiger Abschluss: kein Rebuild noetig; Testprozess danach wieder sauber beendet

## 2026-04-18 Capability Map
- neue kanonische Uebersicht `/workspace/codex/CAPABILITY_MAP.md` erstellt
- produktive Kernpfade, Stub-/Fallback-Bereiche, externe Schnittstellen und Laufzeitabhaengigkeiten dort kompakt zusammengezogen
- kleine Doku-Klarstellung: der aktuelle Polling-Pfad emittiert real `accepted`, `running`, `done`, `failed`; ein separates `queued` wird im aktuellen Code nicht ausgegeben

## 2026-04-18 Director Serve Smoothing
- realen lokalen Director-Umgebungszustand nach dem Restore weiter geglaettet, ohne neuen Feature-Ausbau
- `config/director_llm.env` als echte lokale Default-Konfiguration angelegt und `config/director_llm.env.example` auf denselben Ist-Stand gebracht
- `start.sh`, `init.sh`, `app.main` und `scripts/check_director_llm.py` laden die Director-Defaults jetzt konsistent; optionale lokale Overrides bleiben ueber `config/director_llm.env.local` moeglich
- `scripts/serve_director_llm.sh` um kleine operative Guards erweitert:
  - konfigurierbarer Health-Timeout
  - konfigurierbare Readiness-Retries
  - Bereinigung eines stale PID-Files
  - fruehes Scheitern, wenn `llama-server` vor Readiness beendet wird
- `scripts/check_director_llm.py` um konfigurierbare Timeouts und kleine Retries fuer `/v1/models` und `/v1/chat/completions` erweitert
- `scripts/ensure_llama_cpp.sh` installiert bei Bedarf jetzt auch `ninja`, damit ein Restore-Rebuild nicht am fehlenden Generator haengt
- FastAPI und Director danach erneut sauber als Hintergrunddienste mit PPID `1` gestartet
- neuer echter Verifikationslauf erfolgreich:
  - `POST http://127.0.0.1:8000/agent-core/jobs` mit `director-stability-check-20260418`
  - Director-Pfad lief real ueber `config/director_llm.env` im Modus `llm_augmented`
  - finales MP4 erfolgreich unter `/workspace/agent_runs/director-stability-check-20260418/final.mp4`
  - verifizierte Finaldaten via `ffprobe`: `320x256`, `24 fps`, Gesamtdauer `4.042s`

## 2026-04-18 Restore Startup Hardening
- Restore-/Startup-Zustand nach Repo-Update und Pod-Neustart gegen den kanonischen `/workspace/codex`-Stand geprueft
- bestaetigt: `/workspace/agent_core`, `/workspace/app`, `/workspace/scripts`, `/workspace/config`, `/workspace/tests` und `/workspace/codex` sind vorhanden; die dokumentierten Kernverzeichnisse sind damit wieder vollstaendig
- echter Pod-Startfehler verifiziert: FastAPI scheiterte an `RuntimeError: Directory '/workspace/agent_runs' does not exist`
- minimale Haertung umgesetzt:
  - `app.main` legt die statisch gemounteten Laufzeitordner jetzt vor dem FastAPI-Mount selbst an
  - `start.sh` legt die Basis-Laufzeitordner vor dem Dienststart an
  - `init.sh` legt dieselben Basis-Laufzeitordner idempotent fuer Restore-/Bootstrap-Pfade an
- Regression abgesichert:
  - neuer API-Test fuer die Runtime-Verzeichnis-Erzeugung
  - kompletter Testlauf erneut erfolgreich: `python -m unittest discover -s /workspace/tests -v` -> 49 Tests gruen
- Director-Stack nach Restore real geprueft:
  - GGUF-Modell weiter vorhanden unter `/workspace/models/director/qwen3.6-35b-a3b/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - `llama-server` fehlte nach dem Restore zunaechst real
  - `scripts/serve_director_llm.sh` hat `llama.cpp` daraufhin real neu gebaut und den lokalen Server erfolgreich auf `127.0.0.1:8011` gestartet
  - `scripts/check_director_llm.py` war wegen eines Syntaxfehlers real kaputt und wurde minimal repariert
  - `serve_director_llm.sh` startet jetzt standardmaessig mit `--no-warmup`, passend zum dokumentierten Low-Memory-Profil
- FastAPI live erneut verifiziert:
  - `uvicorn app.main:app` laeuft wieder sauber auf `127.0.0.1:8000`
  - `GET /health` antwortet wieder erfolgreich
  - `GET /agent-core/jobs/does-not-exist` liefert korrekt `404`
  - `GET /agent-core/run` liefert korrekt `405`
- echter Restore-/Startup-Live-Run erfolgreich:
  - `POST http://127.0.0.1:8000/agent-core/jobs` mit `restore-startup-check-20260418`
  - Director real aktiv mit `director_mode=llm_augmented`
  - lokaler Director-Endpoint real genutzt: `http://127.0.0.1:8011/v1/chat/completions`
  - finales MP4 erfolgreich unter `/workspace/agent_runs/restore-startup-check-20260418/final.mp4`
  - verifizierte Finaldaten via `ffprobe`: `320x256`, `24 fps`, Gesamtdauer `4.042s`

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

## 2026-04-17 Phase-5A Director Brain Layer
- kleinste saubere Integrationsstelle im bestehenden Core als Planner-Vorstufe gewaehlt, statt `agent_core` gross zu refactoren
- neue Module `agent_core/director.py`, `agent_core/llm_adapter.py`, `agent_core/prompt_builder.py` und `agent_core/style_memory.py` eingefuehrt
- `ProductionPlan` enthaelt jetzt optional `director_output`; `ScenePlan`, Varianten und Takes dokumentieren jetzt Director-/Prompt-Metadaten explizit
- neues Artefakt `director_output.json` eingefuehrt
- Prompt-Bau fuer Opening-Shots, Stilkonsistenz, visuelle Sprache, Kamera-Hinweise und Variationsabsicht geschaerft
- ehrlichen lokalen OpenAI-kompatiblen Director-Adapter gebaut; ohne produktiven Dienst faellt der Planner klar auf `rule_based_fallback` zurueck
- bestaetigt: im Pod existiert nur `gemma-3` als Teil des LTX2-Stacks, aber kein produktiv laufender lokaler Director-Textdienst; deshalb keine Fake-Gemma-4-Integration gebaut
- `app/agent_core_api.py` im Workspace wiederhergestellt und `app.main` erneut mit `/agent-core` und `/agent-runs` kompatibel gemacht
- neue Tests in `tests/test_director_layer.py` hinzugefuegt
- kompletter Testlauf erneut erfolgreich ausgefuehrt: `python -m unittest discover -s /workspace/tests -v` -> 45 Tests gruen
- echter Live-Fallback-Lauf erfolgreich verifiziert:
  - Job `phase5a-live-fallback-1776420785`
  - Director-Modus `rule_based_fallback`
  - Fallback-Grund `director_llm_not_configured`
  - finales MP4 `/workspace/agent_runs/phase5a-live-fallback-1776420785/final.mp4`

## 2026-04-17 Phase-5B Local Gemma-4 Director Serve
- den bereits vorhandenen Director-Layer auf einen echten lokalen Director-LLM-Pfad umgestellt statt nur auf dokumentierten Fallback
- Pod-Rahmenbedingungen geprueft:
  - nur ca. `33G` frei auf `/workspace`
  - kleinster ehrlicher Produktivpfad ist deshalb GGUF + `llama.cpp`
- neuen lokalen Director-Serve real aufgebaut:
  - `llama.cpp` mit CUDA unter `/workspace/tools/llama.cpp` gebaut
  - `llama-server` produktiv verifiziert
  - `ggml-org/gemma-4-26B-A4B-it-GGUF` als `Q4_K_M` unter `/workspace/models/director/gemma-4-26b-a4b-it/gguf/` angebunden
- neue produktive Helfer eingefuehrt:
  - `scripts/download_director_model.py`
  - `scripts/serve_director_llm.sh`
  - `scripts/check_director_llm.py`
  - `config/director_llm.env.example`
- `agent_core/llm_adapter.py` um das lokale Profil `gemma4_llama_cpp_local` erweitert
- der Director persistiert jetzt explizit `llm_active`, `llm_provider`, `llm_model` und `llm_endpoint`
- fuer den lokalen `llama.cpp`-Pfad wurde der LLM-Request auf einen kleineren `scene_map`-Vertrag umgestellt und im Director danach sauber in den bestehenden `DirectorOutput` normalisiert
- `agent_core/assembler.py` spiegelt den aktiven Director-LLM-Pfad jetzt auch in `result.json`
- kompletter Testlauf erneut erfolgreich: `python -m unittest discover -s /workspace/tests -v` -> 47 Tests gruen
- reale Live-Verifikation:
  - High-Memory-Profil verifiziert Gemma 4 in `llm_augmented`, kollidiert spaeter aber mit LTX2 auf derselben GPU
  - Low-Memory-Profil `-ngl 8 -c 2048 --reasoning off --no-warmup` wurde eingefuehrt
  - echter erfolgreicher Agent-Run `phase5b-live-director-lowmem-1776423376` erzeugt `final.mp4` mit aktivem Gemma-4-Director
  - finaler Agent-Run `phase5b-live-director-final-1776423376` bestaetigt denselben Pfad im aktuellen Codezustand inklusive `result.json`

## 2026-04-18 Phase-5B Switch auf Qwen3.6 Director Serve
- den gestern vorbereiteten Gemma-4-Pfad bewusst nicht weiter ausgebaut, sondern real auf den vom Nutzer priorisierten Qwen3.6-35B-A3B-Pfad umgestellt
- `agent_core/llm_adapter.py` auf das lokale Profil `qwen36_llama_cpp_local` mit Modellstandard `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` umgestellt
- JSON-Parsing im Director-Adapter gehaertet, damit auch Qwen-Antworten mit Begruendungstext oder `<think>` vor dem eigentlichen JSON sauber normalisiert werden
- neue bzw. vervollstaendigte Hilfspfade produktiv genutzt:
  - `config/director_llm.env.example`
  - `scripts/download_director_model.py`
  - `scripts/ensure_llama_cpp.sh`
  - `scripts/serve_director_llm.sh`
  - `scripts/check_director_llm.py`
- reales `llama.cpp`-Binary mit CUDA gebaut und verifiziert:
  - `/workspace/tools/llama.cpp/build/bin/llama-server`
- reale Modellintegration erfolgreich:
  - Download-Quelle: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
  - Modellpfad: `/workspace/models/director/qwen3.6-35b-a3b/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - Groesse im Pod-Lauf: ca. `21.39G`
- ehrliche Download-Notiz:
  - der zuerst angenommene Dateiname ohne `Qwen_`-Praefix existierte nicht und fuehrte zu einem echten `404`
  - danach wurde auf den real vorhandenen Dateinamen `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` umgestellt
- `init.sh` um idempotente Director-Modell-Vorbereitung erweitert:
  - wenn das Qwen-GGUF schon vorhanden ist, wird es nicht neu geladen
  - wenn es fehlt, wird der Download sauber angestossen
  - optional kann der lokale Director-Serve beim Pod-Start automatisch gestartet werden
  - Fehler werden sichtbar geloggt statt still geschluckt
- `app/agent_core_api.py` im Workspace real wiederhergestellt und `app.main` erneut sauber mit `/agent-core` und `/agent-runs` verdrahtet
- Tests erfolgreich:
  - `python -m unittest /workspace/tests/test_director_layer.py -v`
  - `python -m unittest /workspace/tests/test_agent_core_api.py -v`
  - `python -m unittest discover -s /workspace/tests -v` -> 48 Tests gruen
- realer Director-Serve erfolgreich verifiziert:
  - `curl http://127.0.0.1:8011/v1/models`
  - `/workspace/scripts/check_director_llm.py`
- realer erfolgreicher Agent-Run mit aktivem Qwen-Director:
  - Job `phase5b-qwen-live-1776506522`
  - `director_mode=llm_augmented`
  - `director_llm_active=true`
  - `director_llm_provider=llama_cpp_local`
  - `director_llm_model=Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - `director_llm_endpoint=http://127.0.0.1:8011/v1/chat/completions`
  - finales MP4 `/workspace/agent_runs/phase5b-qwen-live-1776506522/final.mp4`

## 2026-04-18 Abschluss-Verifikation und Backup
- gezielte Nachpruefung bestaetigt: die Qwen-Umstellung hat keine Render-Defaults verschoben
- die auffaellige `320x256` aus `phase5b-qwen-live-1776506522` stammt aus einem explizit so gesetzten Verifikationsjob und nicht aus einer neuen Default-Resolution
- bestaetigt: `JobInput.resolution` bleibt standardmaessig `standard`; Landscape-Default bleibt damit `1216x704`
- kleiner Real-Check fuer den idempotenten Modellpfad erfolgreich:
  - `python3 /workspace/scripts/download_director_model.py`
  - Ausgabe: `present: /workspace/models/director/qwen3.6-35b-a3b/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- bestaetigt: `init.sh` startet keinen unnötigen Neu-Download, wenn das Qwen-GGUF bereits vorhanden ist
- kleine Haertung in `init.sh`: vor dem Director-Setup werden alte `director_llm_model_ready`- und `director_llm_server_ready`-Flags geloescht, damit kein veralteter Ready-Status stehenbleibt
- bestaetigt: Fallback-Pfad bleibt im Code- und Testvertrag intakt:
  - `director_llm_not_configured`
  - `director_llm_request_failed:*`
  - Testpfad `tests/test_director_layer.py::test_local_llama_cpp_profile_falls_back_when_server_is_unreachable`
- Backup-Archiv fuer den heutigen uebernehmbaren Stand erzeugt:
  - `/workspace/backups/hyperltx_phase5b_qwen_director_2026-04-18.tar.gz`

## 2026-04-20 Content-Output-Ausbau
- produktiver Music-Step in den bestehenden Core eingebaut:
  - `agent_core/adapters/music_adapter.py` nutzt jetzt real `/Ace_step_1.5`
  - `planner` aktiviert Music nur noch bei real verfuegbarem Backend
  - `agent_core/agent.py` fuehrt Music als echten optionalen Step aus
- `agent_core/assembler.py` auf echten Final-Finish-Pfad erweitert:
  - Voice + Music werden sauber unter explizitem Audio-Mapping gemischt
  - Burn-in-Subtitles und Sidecar-`captions.srt` werden erzeugt
  - optionales Titel-Overlay wird im Final-MP4 eingebrannt
  - finaler Mix bleibt bei genau einem Audio-Stream statt unbeabsichtigter Mehrfach-Audios
- `agent_core/utils.py` um ffmpeg-/Subtitle-Helfer erweitert:
  - Subtitle-Segmentierung und SRT-Schreiben
  - finaler Mix-Renderer
  - spaeter ergaenzter Prompt-Cleanup fuer Storyboard- und Video-Render
- `scripts/agent_core_cli.py` um produktive Flags fuer `--use-music`, `--subtitle-mode`, `--overlay-text`, `--scene-count`, `--variations-per-scene`, `--takes-per-scene` erweitert
- Tests gruen:
  - `python3 -m unittest /workspace/tests/test_planner_rules.py /workspace/tests/test_assembler_mux.py`
- reale Demo-Runs:
  - `demo-social-morning-001`: erster verifizierter Content-Run mit Voice, Music, Storyboard, Burn-in-Subtitles und finalem Mix
  - `demo-social-morning-002`: zweiter echter Vergleichsrun nach Prompt-Bereinigung
- ehrliche Qualitätsnotiz:
  - der Finish-Layer funktioniert real, aber die visuelle Ausgabe hat weiterhin sichtbare Text-/Gibberish-Artefakte im LTX-Bildmaterial
  - Subtitle-Timing stimmt grob, die Segmentierung ist fuer Social-Output aber noch nicht sauber genug
  - der Titel-Overlay ist technisch aktiv, aber bei laengeren Strings noch zu gross und oben angeschnitten

## 2026-04-20 Quality-Fix-First fuer Social-Output
- kleiner produktiver Prompt-/Caption-/Overlay-Pass statt weiterem Kernumbau umgesetzt
- `agent_core/utils.py` erweitert um:
  - haerteres `compress_visual_prompt(...)` gegen Narrationssatztext und Text-/UI-/Dokument-Artefakte
  - gezielte Sanitizer fuer textanfaellige Papier-/Notizbuch-Phrasen
  - Merge-/Mindestdauer-Regeln fuer Subtitle-Segmente
  - Auto-Wrap und Layout-Profil fuer Titel-Overlay
- `agent_core/assembler.py` nutzt jetzt:
  - neue Subtitle-Parameter fuer Minimum-Dauer und Short-Merge
  - vorformatierte Overlay-Titeldateien statt ungewrappter Rohstrings
- `agent_core/planner.py` haertet Storyboard-Keyframe-Prompts zusaetzlich gegen Text-/Dokument-Artefakte
- neuer Utility-Test `tests/test_output_quality_utils.py`
- Tests gruen:
  - `python3 -m unittest /workspace/tests/test_output_quality_utils.py /workspace/tests/test_assembler_mux.py /workspace/tests/test_planner_rules.py`
- reale Vergleichslaeufe:
  - `demo-social-morning-003`: Overlay-Clipping behoben, Caption-Split besser, fruehe Textartefakte klar reduziert; spaeter Frame weiterhin Papier-Artefakte
  - `demo-social-morning-004`: zweiter echter Nachfix-Run; spaeter Payoff-Frame deutlich sauberer, aber Schreibszene wieder mit starkem Dokument-/Gibberish-Muell
- ehrlicher Stand:
  - Overlay-Layout jetzt robust genug fuer typische Social-Titel
  - Subtitle-Segmentierung sichtbar besser als `demo-social-morning-002`
  - Anti-Text-Steuerung bleibt modellseitig inkonsistent und ist der groesste verbleibende Qualitaetsengpass

## 2026-04-20 Social-Tipp-Format-Guard
- enger produktiver Guard in `agent_core/planner.py` eingebaut statt weiterer allgemeiner Anti-Text-Experimente
- kurze Portrait-Voice-Social-Clips mit Storyboard/Music/Subtitles werden jetzt auf robuste Motivklassen eingeschraenkt
- explizit vermiedene Planner-Motive:
  - Papier, Notizbuch, Dokumente, Seiten, Handschrift, Schreiben, Label, Signs, Posters, UI, App-Screens, Monitor-Closeups, Buchseiten, Sticky Notes, Printed Notes
- bevorzugte Guard-Motive:
  - Aufwachen + Vorhaenge
  - Fensterlicht + Stretch
  - Glas Wasser + Handy face-down
  - ruhiges neutrales B-Roll
  - Window-/Coffee-/Breathing-Payoff
- neuer Test in `tests/test_planner_rules.py` verifiziert, dass textnahe Schreib-/Papiermotive fuer dieses Format nicht mehr im Planner landen
- Tests gruen:
# 2026-05-05 Creative OS CLI Cockpit V1.6

- Rich Cockpit Grid fuer `scripts/creative_os_status.py --style rich` finalisiert.
- Layout: kompakter Header, linke Sidebar mit System Status und Pipeline Map, dominante Active Workspace Flaeche, Scene Jobs und Bottom Grid.
- `--style plain` bleibt Default und stabil.
- Snapshots erzeugt unter `/workspace/cli_cockpit_snapshots/`.
- Tests: `python3 -m unittest /workspace/tests/test_creative_os_status.py -v` -> 12 Tests OK.
- Kein Stage 8, kein Render, kein LTX, kein Video, kein Backend-Aufruf, kein n8n, keine API, kein Textual.

  - `python3 -m unittest /workspace/tests/test_planner_rules.py /workspace/tests/test_output_quality_utils.py /workspace/tests/test_assembler_mux.py`
- realer E2E-Run:
  - `demo-social-morning-005`
  - `director_mode=llm_augmented`
  - `success=true`
  - visuell klar sauberer als `demo-social-morning-004`, weil keine textnahen Papier-/Notizszenen mehr auftauchen
- ehrliche Restrisiko-Notiz:
  - trotz robusterer Motive bleiben kleine runabhaengige Glyph-/Textfragmente im Modelloutput moeglich

## 2026-04-21 Narrow Social Quality Pass

- produktiver Social-Tipp-Guard in `agent_core/planner.py` zu einer kleinen Motivbibliothek erweitert:
  - `morning_reset`
  - `focus_break`
  - `kitchen_reset`
  - `movement_reset`
- produktiver Subtitle-Hebel nachgezogen:
  - Social-Tipp-Plaene setzen jetzt `subtitle_min_words=3`
  - Social-Tipp-Plaene setzen jetzt `subtitle_min_duration_sec=1.1`
  - `agent_core/assembler.py` liest diese Subtitle-Defaults jetzt aus `plan.metadata`
- neue Tests:
  - `tests/test_planner_rules.py` verifiziert Family-Zuordnung und display-fernen `focus_break`-Style-Lock
  - `tests/test_assembler_mux.py` verifiziert, dass der Assembler die produktiven Subtitle-Defaults aus dem Plan uebernimmt
- Tests gruen:
  - `python3 -m unittest /workspace/tests/test_planner_rules.py /workspace/tests/test_output_quality_utils.py /workspace/tests/test_assembler_mux.py`
- realer Kontrollbefund:
  - `demo-social-morning-006` lief noch ueber stale Uvicorn-Live-Code und zeigte in `plan.json` weiter `social_tip_visual_guard_version=v1`
  - daraus folgte ein minimaler produktiver Uvicorn-Neustart auf `8000`, weil der Pod-Server ohne Auto-Reload laeuft
- reale Nachverifikation Morning:
  - `demo-social-morning-007`
  - `director_mode=llm_augmented`
  - `final.mp4` real vorhanden
  - `plan.json` zeigt `social_tip_visual_guard_version=v2`, Family `morning_reset` und die neue Kuechenroutine in Szene 3
  - sichtbarer Befund: kohaerenter und social-lesbarer als die dokumentierte `demo-social-morning-005`-Basis, aber weiter mit kleinem Glyph-/Textmuell im Payoff
- reale Nachverifikation Focus-Break:
  - `demo-social-focus-break-001` zeigte trotz neuer Szenenfolge starkes Whiteboard-/Screen-/Textmuell
  - daraufhin minimaler Nachfix in `agent_core/planner.py`: Social-Tipp-Familien ueberschreiben jetzt auch `style_lock.visual_identity`
  - `demo-social-focus-break-002` lief danach real mit display-fernem Style-Lock und `final.mp4`
  - ehrlicher Sichtbefund: leicht entschärft, aber weiterhin klar ungenuegend; Office-/Papier-/Screen-/Glyph-Artefakte bleiben fuer diese Familie offen
# 2026-05-03 G6 Skill Injection
- `agent_core/creative_system/skill_injection.py` eingefuehrt.
- Model-Skills fuer Z-Image, LTX und Qwen3-VL Review ergaenzt.
- Agent-Skill-Trace auf zentralen SkillInjectionContext umgestellt.
- Stage Contracts um SkillInjectionContext, aktivere Skill-IDs und Review-Kriterien erweitert.
- PromptBuilder-Trace um `ltx_positive_prompt_sent` und `ltx_negative_prompt_sent` erweitert.
- `prompt_audit.json`, `model_prompts.json`, `stage_contracts.json` und `decision_log.json` enthalten jetzt skill-/contract-basierte G6-Daten.
- Qwen3-VL Reviewer-Systemprompt um low phone-size readability und stabilen JSON-Vertragshinweis erweitert.
- Neue Tests: `tests/test_g6_skill_injection.py`, `tests/test_g6_promptbuilder_skill_policy.py`, `tests/test_g6_review_skill_policy.py`.
- Sicherer Smoke: `/workspace/agent_runs/g6-skill-injection-stop-after-model-prompts-smoke`, kein `final.mp4`.

# 2026-05-03 G7 Creative Beat Planner
- `agent_core/creative_system/strategy_planner.py` eingefuehrt.
- `clean_shortform_v1` erzeugt jetzt `CreativeIntent`, mindestens drei `BeatPlanCandidate`-Varianten, Score-Breakdowns und eine selected Candidate Decision.
- Planner nutzt den selected Candidate fuer ScenePlan und per-scene VisualDirection.
- PromptBuilder nutzt per-scene Direction, selected motif/shot recipe und Candidate-Kontext fuer model-facing Prompts.
- Stage Contracts, Prompt Audit, Model Prompts und DecisionLog enthalten G7-Trace: Intent, Candidates, Scores, selected Candidate und per-scene Direction.
- G6-Polish: Script-Literal-Leakage in visual_goal/model prompts wird systemisch ueber Sanitizing und semantische per-scene Actions vermieden; alte Morning-Reset-Motive bleiben Kandidaten, aber nicht Pflichtsequenz im G7-Pfad.
- `agent_core/feedback_policy.py` als G8 Scaffold eingefuehrt; Review-Issues werden zu FeedbackAction-Vorschlaegen gemappt, noch ohne Executor.
- Neue Tests: `tests/test_g7_creative_intent.py`, `tests/test_g7_beat_plan_candidates.py`, `tests/test_g7_planner_integration.py`, `tests/test_g8_feedback_policy.py`.
- Sicherer Smoke: `/workspace/agent_runs/g7-beat-planner-stop-after-model-prompts-smoke`, gestoppt bei `model_prompts`, kein `final.mp4`.

# 2026-05-03 G8 Feedback Loop / Retry Policy Scaffold
- `agent_core/feedback_policy.py` erweitert um `FeedbackAction`, `RetryBudget`, `RetryPlan`, `evaluate_feedback_actions`, `build_retry_plan` und checkpoint-kompatible State-Ausgabe.
- Issue-Mapping deckt sichtbaren Text, Fake-Text, UI/Device, boring/dead/static, weak hook, unclear action, generic stock feel, physical incoherence, low phone-size readability, voice/visual mismatch und bad composition ab.
- DecisionLog kann `feedback_action_created`, `retry_plan_created`, `blocked_by_feedback`, `human_review_required` und `artifact_invalidated` speichern.
- CLI `--inspect-run` zeigt vorhandene `feedback_actions.json` und `retry_plan.json` inklusive Top Action, Suggested Fix, Blockierung und Invalidations an.
- Sicheres Fixture: `/workspace/agent_runs/g8-feedback-policy-smoke`, kein `final.mp4`.
- Neue Tests: `tests/test_g8_feedback_actions.py`, `tests/test_g8_retry_plan.py`, `tests/test_g8_cli_feedback_inspect.py`.
- Kein echter Retry Executor, kein Render, keine Modelle, keine Downloads, keine Runtime-/Docker-/`init.sh`-/Backend-/n8n/API/GUI-Aenderungen.

# 2026-05-03 G9 First V1 Run
- Preflight-Tests fuer G6/G7/G8/G1 waren gruen.
- Dry-Run `g9-v1-morning-reset-dryrun-001` stoppte sauber bei `model_prompts`; kein `final.mp4`.
- Dry-Run zeigte `creative_intent`, 3 BeatPlanCandidates, selected Candidate `tactile_first`, per-scene VisualDirection, Z-Image positive-only und LTX positive/negative Trace.
- Genau ein echter Render wurde gestartet: `g9-v1-morning-reset-render-001`.
- Render-Konfiguration: `clean_shortform_v1`, portrait `512x768`, 3 Szenen, Storyboard true, LTX, no voice, no music, subtitles off, 1 Variation, 1 Take, heuristic review.
- Resultat: technischer Success, `final.mp4` vorhanden, Final Quality Verdict `needs_review`, `real_vlm_inference_used=false`.
- Manueller Frame-Befund: Szene 2 enthaelt sichtbare text-/UI-/Papierartefakte; Clip ist interner Systembeweis, aber nicht demo-wuerdig.
- G8 FeedbackPolicy erzeugte `feedback_actions.json` und `retry_plan.json`; Top Action `visible_text -> regenerate_keyframe`, blocking true.
- Kein Retry-Render gestartet.
