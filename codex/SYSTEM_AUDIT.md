# SYSTEM_AUDIT.md

## Scope
Stand der technischen Bestandsaufnahme am 2026-04-11 in der laufenden RunPod-Umgebung.

## 1. Verifizierte Fakten

### Umgebung
- Host: Ubuntu 22.04-basierte CUDA-Containerumgebung
- User: `root`
- Kernel: `Linux 6.8.0-90-generic`
- Python im Root-Environment: `3.12.13`
- `pip`: `26.0.1`
- Aktive Root-Venv: keine (`VIRTUAL_ENV` leer)
- CUDA Driver laut `nvidia-smi`: `570.211.01`
- CUDA Toolkit: `12.8` vorhanden, `nvcc` unter `/usr/local/cuda/bin/nvcc`
- GPU: `NVIDIA RTX 6000 Ada Generation`
- Torch-Probe: `torch 2.7.0+cu128`, CUDA verfuegbar, 1 GPU erkannt
- RunPod-Metadaten vorhanden:
  - `RUNPOD_GPU_COUNT=1`
  - `RUNPOD_CPU_COUNT=10`
  - `RUNPOD_MEM_GB=167`
  - `RUNPOD_POD_ID` gesetzt
- `/workspace` ist eigenes Dateisystem mit ca. `160G`, davon ca. `128G` belegt und ca. `33G` frei

### Python-Runtimes
- Root-Python:
  - `fastapi 0.135.3`
  - `uvicorn 0.44.0`
  - `pydantic 2.12.5`
  - `diffusers 0.37.1`
  - `transformers 4.52.4`
  - `accelerate 1.12.0`
  - `bitsandbytes 0.49.2`
  - `librosa 0.11.0`
  - `soundfile 0.13.1`
  - `av 17.0.0`
  - `torch 2.7.0+cu128`
  - `torchaudio 2.7.0+cu128`
  - `torchvision 0.22.0+cu128`
- Separates Venv: `/workspace/venvs/qwen3-tts`
  - Python `3.12.13`
  - `transformers 4.57.3`
  - `accelerate 1.12.0`
  - `diffusers 0.37.1`
  - `modelscope 1.35.4`
  - `qwen-tts 0.1.1`
  - `nano-vllm 0.2.0`
  - `torch 2.7.0+cu128`
  - `torchaudio 2.7.0+cu128`
  - `torchvision 0.22.0+cu128`

### Vorhandene CLI-Tools
- Verifiziert vorhanden:
  - `ffmpeg`
  - `ffprobe`
  - `sox`
  - `git`
  - `curl`
  - `node`
  - `npm`
  - `uvicorn`
  - `jupyter`
- Verifiziert nicht vorhanden:
  - `wget`
  - `jq`
  - `ss`

### Laufende Dienste
- Prozess `jupyter-lab` laeuft ueber `/workspace/start.sh`
- Prozess `uvicorn app.main:app --host 0.0.0.0 --port 8000` laeuft
- Healthcheck erfolgreich:
  - `GET /health` -> `{"status":"ok","init_ready":true,"ltx_backend":"ltx-2.3"}`
- Readiness erfolgreich:
  - `GET /DW/ready` -> `ready=true`
  - `GET /DW/qwen_tts_ready` -> `ready=true`
  - `GET /DW/ace_step_1_5_ready` -> `ready=true`
  - `GET /DW/zimage_ready` -> `ready=true`

### Vorhandene Services, APIs und Skripte
- Einstiegspunkt: `/workspace/app/main.py`
- Vorhandene API-Flaechen:
  - `/health`
  - `/DW/*` Readiness-Endpunkte
  - `/ltx2/*`
  - `/qwen_tts/*`
  - `/Ace_step_1.5/*`
  - `/zimage/*`
  - `/editor/render`
  - `/upscale/*`
- Start-/Init-Skripte:
  - `/workspace/start.sh`
  - `/workspace/init.sh`
  - `/workspace/logs.sh`
  - `/workspace/upscaler_installer_minimal/install_realesrgan_ai_pod.sh`
  - `/workspace/upscaler_installer_minimal/upscale_video_ai_cuda.sh`

### Repo-Lage
- Git-Root: `/workspace`
- Remote: `origin -> https://github.com/Leon102938/HyperLTX-2`
- Aktueller Branch: `main`
- Letzter sichtbarer Commit: `8d2f249 LTX2_Fix`
- Die getrackten Inhalte konzentrieren sich auf:
  - `ACE-Step-1.5` sehr gross
  - `LTX-2`
  - `Qwen3-TTS`
  - `app`
  - Init-/Startskripte und Konfiguration

### Repo-Ueberblick
- `/workspace/agent_core`
  - neuer modularer Phase-1-Agent-Core mit Agent, Planner, State-Store, Registry, Assembler und Adaptern
- `/workspace/app`
  - FastAPI-Wrapper und lokale Job-Logik fuer Medien-Backends
- `/workspace/LTX-2`
  - eingebettetes LTX-2-Monorepo mit `ltx-core`, `ltx-pipelines`, `ltx-trainer`
- `/workspace/ACE-Step-1.5`
  - eingebettetes Musik-Repo mit API-, CLI- und Trainingsbereichen
- `/workspace/Qwen3-TTS`
  - lokaler Qwen3-TTS-Code inklusive Finetuning-Unterordner
- `/workspace/models/qwen3-tts`
  - lokale Qwen3-TTS-Modelle
- `/workspace/jobs`
  - Jobordner fuer Backends
- `/workspace/exports`
  - exportierte Ausgaben
- `/workspace/status`
  - Readiness-Flags
- `/workspace/venvs/qwen3-tts`
  - Spezial-Venv fuer Qwen/ACE-Step-nahe Runtimes

### Vorhandene Modell- und Output-Strukturen
- LTX-2:
  - `/workspace/LTX-2/checkpoints/ltx-2.3/ltx-2.3-22b-dev.safetensors`
  - `/workspace/LTX-2/checkpoints/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors`
  - `/workspace/LTX-2/checkpoints/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
  - `/workspace/LTX-2/checkpoints/gemma-3/*`
- Qwen TTS:
  - `/workspace/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `/workspace/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz`
- ACE-Step:
  - `/workspace/ACE-Step-1.5/checkpoints/acestep-v15-turbo`
  - `/workspace/ACE-Step-1.5/checkpoints/vae`
  - `/workspace/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B`
  - `/workspace/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B`

### Projektgedaechtnis
- In der Umgebung existierte bereits ein Jupyter-erstellter Projekt-Memory-Stand unter `/workspace/Codex`.
- Der vom Nutzer geforderte kanonische Pfad `/workspace/codex` wurde fuer die fortlaufende Pflege angelegt.

### Neu hinzugefuegte Core-Bausteine
- `agent_core/agent.py`
- `agent_core/schemas.py`
- `agent_core/planner.py`
- `agent_core/state_store.py`
- `agent_core/backend_registry.py`
- `agent_core/assembler.py`
- `agent_core/utils.py`
- `agent_core/adapters/*`
- `examples/minimal_job.json`
- `tests/test_core_smoke.py`
- `tests/test_planner_rules.py`

## 2. Annahmen
- Der aktuelle FastAPI-Layer ist eher ein bestehender Backend-Wrapper als der gewuenschte neue Agent-Core.
- Die vorhandenen lokalen Python-Module sind fuer Phase 1 die sinnvollste Integrationsflaeche.
- Die vorhandenen Readiness-Flags spiegeln Modell- und Runtime-Verfuegbarkeit wider, nicht die Existenz eines orchestrierenden Cores.

## 3. Offene Fragen
- Welcher Minimal-Flow soll den ersten Agent-Core definieren?
- Soll der erste Core lokal direkt Python-Backends aufrufen oder deren vorhandene API-Flaechen konsumieren?
- Welcher Output gilt in Phase 1 als Erfolg: fertiges MP4, nur Job-State plus Referenz auf Backend-Output, oder beides?
- Welche Jobparameter muessen bereits in Phase 1 stabil sein und welche duerfen vorerst fehlen?
- Soll das bestehende Repo spaeter intern umstrukturiert werden oder bleibt der Core bewusst als duenne Ebene darueber?

## 4. Risiken
- Root-Python und `qwen3-tts`-Venv haben unterschiedliche Paketstaende; unsaubere Adapter koennen in die falsche Runtime laufen.
- Nur ca. `33G` Speicher frei; grosse weitere Modell-Downloads sind risikobehaftet.
- Secret-behaftete Umgebungsvariablen sind im Prozessumfeld vorhanden; Logs und Doku duerfen keine Werte leaken.
- Der aktuelle Repo-Zustand enthaelt bereits mehrere grosse eingebettete Subrepos; tiefe Umbauten wuerden Phase 1 verlangsamen.
- Bisher fehlt ein einheitlicher Core-State fuer zusammengesetzte Jobs ueber mehrere Backends.

## 5. Luecken
- kein vollstaendig real verifizierter Heavy-Run des neuen Core ueber Qwen TTS plus LTX2 in dieser Session
- noch keine produktiven Integrationen fuer Music, Storyboard, ZImage, ACE-Step oder Editor/Upscaler im neuen Core
- noch keine externe API-Schicht fuer `agent_core`
- noch keine n8n-Anbindung fuer `agent_core`

## 6. Spaeter relevante Referenzen
- Lokales Haupt-Repo: `https://github.com/Leon102938/HyperLTX-2`
  - bereits lokal als Git-Root unter `/workspace`
- Spaeteres Referenz-Repo: `https://github.com/Matticusnicholas/KupkaProd-Cinema-Pipeline`
  - aktuell nicht lokal vorhanden
  - wurde in diesem Recon nicht analysiert
  - moegliche spaetere Nutzung: Ideen fuer Pipeline-Organisation, Render-/Assembly-Muster oder Produktions-Workflows
  - Einbindungspaetere Option: separat klonen oder nur gezielt als Referenz spiegeln, ohne blindes Uebernehmen

## 7. Empfehlungen fuer Phase 1
- Einen kleinen Core als neue, eigenstaendige Schicht anlegen.
- Vorhandene Backends ueber duenne Adapter kapseln.
- Zuerst nur einen linearen Minimal-Flow unterstuetzen.
- Job- und Step-State als JSON im Dateisystem persistieren.
- Den ersten Smoke-Test mit bereits verifizierten lokalen Assets bauen.
- API, n8n, GUI, grosse Refactors und neue Modell-Downloads explizit aus Phase 1 heraushalten.
