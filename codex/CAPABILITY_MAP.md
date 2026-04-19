# CAPABILITY_MAP.md

## A) Aktueller Gesamtstatus
- Abgeschlossen: Phase 1, 2A, 2B, 2C, 2D, 2E, 3A, 3B, 4A, 4B, 4C, 5A, 5B
- Produktiv laufende Hauptbloecke:
  - `agent_core` als lokaler Produktionskern
  - FastAPI-Bridge auf Port `8000`
  - LTX2-Video ueber lokale FastAPI-Endpunkte
  - Qwen-TTS-Voice ueber lokale FastAPI-Endpunkte
  - Z-Image-Keyframes ueber lokale FastAPI-Endpunkte
  - lokaler Qwen3.6-Director ueber `llama.cpp` auf Port `8011`
  - `final.mp4`-Assembly mit persistierten Artefakten unter `/workspace/agent_runs/<job_id>/`

## B) Produktiv funktionierende Kernpfade
- Async API bridge: `POST /agent-core/jobs` startet reale Jobs nicht blockierend.
- Polling path: `GET /agent-core/jobs/{job_id}` ist der produktive Statuspfad fuer externe Caller.
- n8n-friendly polling: `status_summary`, `is_terminal`, `should_poll`, `retry_after_sec`, `artifacts_ready`, `final_mp4_ready`, `result_json_ready`, `public_refs` sind real im Vertrag.
- Director Qwen path: lokaler `llama.cpp`-Serve mit GGUF ist real verifiziert und liefert `llm_augmented`.
- Director fallback: ohne nutzbaren Director bleibt `rule_based_fallback` sauber aktiv und wird persistiert.
- Multi-scene: mehrere Szenen pro Job mit `scene_plan.json` und Concat vor Finalisierung.
- Multi-take: mehrere Takes pro Szene mit persistierter Auswahl in `takes.json`.
- Quality guard: technische Video-Validierung vor Take-Selektion ist produktiv aktiv.
- Storyboard/keyframe: Z-Image erzeugt reale Keyframe-Kandidaten; Auswahl wird persistiert.
- Keyframe-conditioned video: selektierte Storyboard-Keyframes koennen im stabilen `ti2vid`-Pfad als First-Frame-Conditioning genutzt werden.
- Voice / Qwen TTS: echte WAV-Erzeugung ueber Qwen3-TTS ist produktiv verifiziert.
- Final assembly: `final.mp4` wird per Assembler erzeugt; Voice wird gemuxt oder das Video sauber gespiegelt.

## C) Adapter- und Modulstatus

| Pfad | Status | Kurznotiz |
|---|---|---|
| `/workspace/agent_core/agent.py` | `productive` | Orchestriert Planner, optionale Voice, optionales Storyboard, Video, Assembly und Persistenz. |
| `/workspace/agent_core/planner.py` | `productive` | Baut den quantisierten Produktionsplan fuer Single- und Multi-Scene-Jobs. |
| `/workspace/agent_core/assembler.py` | `productive` | Baut `final.mp4`, muxed Voice oder spiegelt das Video ohne Voice. |
| `/workspace/agent_core/state_store.py` | `productive` | Kanonische Persistenz fuer `input_job.json`, `plan.json`, `state.json`, `result.json`, Logs und Reports. |
| `/workspace/agent_core/backend_registry.py` | `productive` | Registriert die aktuell real genutzten Adapter und waehlt den primaeren Adapter pro Kind. |
| `/workspace/agent_core/adapters/ltx2_adapter.py` | `productive` | Produktiver Video-Adapter gegen lokale `/ltx2/*`-Endpoints; stabiler Kernpfad ist `ti2vid`. |
| `/workspace/agent_core/adapters/qwen_tts_adapter.py` | `productive` | Produktiver Voice-Adapter gegen lokale `/qwen_tts/*`-Endpoints. |
| `/workspace/agent_core/adapters/zimage_storyboard_adapter.py` | `productive` | Produktiver Storyboard-/Keyframe-Adapter gegen lokale `/zimage/*`-Endpoints. |
| `/workspace/agent_core/adapters/storyboard_adapter.py` | `stub` | Reiner Stub-Adapter; nicht produktiv genutzt. |
| `/workspace/agent_core/adapters/music_adapter.py` | `stub` | Musik-Schnittstelle nur vorbereitet; kein produktiver Musikpfad im Core. |
| `/workspace/agent_core/director.py` | `productive` | Director-/Brain-Schicht vor dem Planner; normalisiert LLM- oder Fallback-Ausgabe in den bestehenden Vertrag. |
| `/workspace/agent_core/llm_adapter.py` | `productive` | Nutzt den lokalen OpenAI-kompatiblen Director-Endpoint oder faellt ehrlich auf Fallback zurueck. |
| `/workspace/agent_core/prompt_builder.py` | `productive` | Baut produktive Prompt-/Variationsbausteine fuer Director- und Planner-Pfad. |
| `/workspace/agent_core/style_memory.py` | `productive` | Baut den aktuellen `style_lock`; kein tiefer persistenter Character-/World-Lock. |
| `/workspace/app/agent_core_api.py` | `productive` | Produktive Bridge fuer `POST /agent-core/jobs` und `GET /agent-core/jobs/{job_id}`; `POST /agent-core/run` bleibt Dev/Test. |
| `/workspace/app/main.py` | `productive` | Zentraler FastAPI-Entry, Mounts, Readiness und LTX2-Endpoints. |
| `/workspace/app/LTX2.py` | `productive` | Reales lokales LTX2-Backend fuer den Core. |
| `/workspace/app/qwen_tts.py` | `productive` | Reales lokales Qwen-TTS-Backend fuer den Core. |
| `/workspace/app/zimage.py` | `productive` | Reales lokales Z-Image-Backend fuer Storyboard-/Keyframe-Jobs. |
| `/workspace/app/ace_step_1_5.py` | `partial` | Backend existiert und hat Endpoints, ist aber nicht in den aktuellen produktiven Core-Pfad integriert. |
| `/workspace/app/editor_api.py` | `legacy` | Verfuegbar, aber nicht Teil des aktuellen `agent_core`-Produktivpfads. |
| `/workspace/app/upscaler_api.py` | `legacy` | Verfuegbar, aber nicht Teil des aktuellen `agent_core`-Produktivpfads. |
| `/workspace/start.sh` | `productive` | Startet Basisdienste und legt Laufzeitordner vor Dienststart an. |
| `/workspace/init.sh` | `productive` | Richtet Modelle, Venvs, Flags und optional den Director-Serve idempotent ein. |
| `/workspace/scripts/download_director_model.py` | `productive` | Minimaler Download-/Present-Check fuer das lokale Director-GGUF. |
| `/workspace/scripts/serve_director_llm.sh` | `productive` | Produktiver lokaler Director-Serve inkl. Build-/Health-/Readiness-Guards. |
| `/workspace/scripts/check_director_llm.py` | `productive` | Produktiver Smoke-Check fuer `/v1/models` und `/v1/chat/completions`. |
| `/workspace/scripts/ensure_llama_cpp.sh` | `productive` | Baut `llama.cpp` bei Bedarf real neu. |
| `/workspace/scripts/agent_core_cli.py` | `productive` | Kleines lokales Manual-Test-CLI auf dem bestehenden `POST /agent-core/jobs` plus Polling-Vertrag. |
| `/workspace/config/director_llm.env` | `productive` | Reale lokale Default-Konfiguration fuer den Director-Pfad. |
| `/workspace/config/director_llm.env.example` | `partial` | Vorlage, nicht selbst der aktive Laufzeitpfad. |
| `/workspace/codex` | `productive` | Kanonisches Projektgedaechtnis und Doku-Quelle. |

## D) Externe Schnittstellen

### Produktiv fuer externe Caller
- `POST /agent-core/jobs`
  - bevorzugter produktiver Einstieg fuer Automation und n8n-nahe Caller
- `GET /agent-core/jobs/{job_id}`
  - bevorzugter Polling-Pfad
  - reale Statuswerte aktuell: `accepted`, `running`, `done`, `failed`
  - wichtig: ein separates `queued` wird vom aktuellen Code nicht emittiert
- `GET /agent-runs/{job_id}/state.json`
- `GET /agent-runs/{job_id}/result.json`
- `GET /agent-runs/{job_id}/final.mp4`
- `GET /health`

### Produktiv als interne Backend-Schnittstellen
- LTX2:
  - `POST /ltx2/submit`
  - `GET /ltx2/status/{job_id}`
  - `GET /ltx2/get/{job_id}`
- Qwen TTS:
  - `POST /qwen_tts/custom_voice`
  - `GET /qwen_tts/status/{job_id}`
  - `GET /qwen_tts/get/{job_id}`
- Z-Image:
  - `POST /zimage/jobs`
  - `GET /zimage/jobs/{job_id}`
  - `GET /zimage/jobs/{job_id}/file`
- Readiness:
  - `GET /DW/ready`
  - `GET /DW/qwen_tts_ready`
  - `GET /DW/zimage_ready`
  - `GET /DW/ace_step_1_5_ready`

### Nur Dev/Test oder nicht Teil des bevorzugten Core-Pfads
- `POST /agent-core/run`
  - synchroner Dev-/Test-Pfad
- `POST /editor/render`
- `POST /upscale/video`
- `POST /upscale/submit`
- `GET /upscale/get/{job_id}`
- `GET /upscale/log/{job_id}`
- `POST /Ace_step_1.5/generate`
- `GET /Ace_step_1.5/status/{job_id}`
- `GET /Ace_step_1.5/get/{job_id}`

## E) Laufzeit- und Abhaengigkeitslogik
- Erwartete produktive Laufzeitbausteine:
  - Python/FastAPI/uvicorn
  - `ffmpeg` und `ffprobe`
  - `sox`
  - CUDA-faehiges Torch
  - LTX2-Checkpoints unter `/workspace/LTX-2/checkpoints`
  - Qwen-TTS-Modelle unter `/workspace/models/qwen3-tts`
  - Qwen-TTS-Venv unter `/workspace/venvs/qwen3-tts`
  - Director-GGUF unter `/workspace/models/director/qwen3.6-35b-a3b/gguf`
  - `llama.cpp` unter `/workspace/tools/llama.cpp`
- Durch `init.sh` bzw. Scripts vorbereitet:
  - Basis-Laufzeitordner
  - Qwen-TTS-Venv und Modelle
  - ACE-Step-Shared-Runtime
  - Director-GGUF
  - optionaler lokaler Director-Serve
- Nicht im normalen Repo-Backup enthalten, aber fuer reale Runs wichtig:
  - `/workspace/LTX-2/checkpoints`
  - `/workspace/models/qwen3-tts`
  - `/workspace/models/director`
  - `/workspace/tools/llama.cpp`
  - `/workspace/venvs`
  - `/workspace/agent_runs`
  - `/workspace/jobs`
  - `/workspace/exports`
  - `/workspace/status`

## F) Noch nicht fertig / naechste Luecken
- Kein produktiver Musikpfad im `agent_core`; `music_adapter.py` bleibt Stub, obwohl `ACE-Step 1.5` als Backend ausserhalb des Core verfuegbar ist.
- Kein zweiter stabiler Video-Produktivpfad; `a2vid` ist nicht freigegeben.
- Keine durable Queue, keine restart-sichere Worker-Schicht, keine echte n8n-Orchestrierung.
- Keine Auth-, Multi-User- oder GUI-Schicht.
- Director ist real produktiv, aber bisher nur in kleinen bzw. gezielten Live-Runs verifiziert; weitere echte Multi-Scene-/Storyboard-Live-Validation fehlt noch.
- `style_memory.py` ist funktional, aber noch kein tiefer Character-/Voice-/World-Lock ueber laengere Produktionen.
- Keyframe-Conditioning ist produktiv nur als First-Frame-Pfad; keine Multi-Keyframe-Interpolation, kein Retake-System fuer Keyframes.

## Testabdeckung
- `tests/test_core_smoke.py`: Kernlauf und Basispfade
- `tests/test_planner_rules.py`: Planner- und Vertragsregeln
- `tests/test_assembler_mux.py`: Final-MP4-Assembly
- `tests/test_scene_planner.py`: Multi-Scene und Varianten
- `tests/test_take_quality_guard.py`: Quality-Guard und Retry-Logik
- `tests/test_storyboard_pipeline.py`: Storyboard-/Keyframe-Pfad
- `tests/test_director_layer.py`: Director-/LLM-/Fallback-Vertrag
- `tests/test_agent_core_api.py`: FastAPI-Bridge und Polling-Vertrag
