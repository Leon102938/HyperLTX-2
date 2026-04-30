# CHANGELOG.md

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
