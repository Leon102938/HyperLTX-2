# COMMAND_PROMPTS.md

## Zweck
Wiederverwendbare Shell-Befehle, Checks und Prompt-Bausteine fuer dieses Projekt.

## Shell: Schnellchecks

### Umgebung
```bash
date -u '+%Y-%m-%dT%H:%M:%SZ'
python --version
which python
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
```

### Dienste
```bash
ps -eo pid,ppid,cmd --sort=pid | sed -n '1,200p'
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/DW/ready
curl -sS http://127.0.0.1:8000/DW/qwen_tts_ready
curl -sS http://127.0.0.1:8000/DW/ace_step_1_5_ready
curl -sS http://127.0.0.1:8000/DW/zimage_ready
```

### Repo-Recon
```bash
git -C /workspace status --short
git -C /workspace remote -v
git -C /workspace log --oneline -n 10
find /workspace -maxdepth 2 -type d | sort
rg -n '@app\\.|@router\\.' /workspace/app
```

### Commit-Hygiene pruefen
```bash
git -C /workspace status --short
git -C /workspace diff -- .gitignore
find /workspace/agent_core -type d \\( -name __pycache__ -o -name .ipynb_checkpoints \\) -prune
find /workspace/tests -type d -name __pycache__ -prune
```

### Modelle und Speicher
```bash
du -sh /workspace/LTX-2/checkpoints /workspace/ACE-Step-1.5/checkpoints /workspace/models/qwen3-tts
df -h /workspace /
find /workspace/status -maxdepth 2 | sort
```

### Qwen-Venv pruefen
```bash
/workspace/venvs/qwen3-tts/bin/python --version
/workspace/venvs/qwen3-tts/bin/pip list | rg '^(torch|transformers|accelerate|diffusers|qwen-tts|modelscope)\\b'
```

### Agent-Core testen
```bash
python -m unittest discover -s /workspace/tests -v
python - <<'PY'
from agent_core.agent import VideoAgent
result = VideoAgent().run_job('/workspace/examples/minimal_job.json')
print(result.model_dump())
PY
```

### Finales MP4 pruefen
```bash
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,duration -of json /workspace/agent_runs/<job_id>/final.mp4
python - <<'PY'
import json, pathlib
job_id = "real-e2e-mux-2"
base = pathlib.Path("/workspace/agent_runs") / job_id
print(base.joinpath("result.json").read_text())
print(base.joinpath("state.json").read_text())
PY
```

### Dauervertrag pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-duration-case-a"
base = pathlib.Path("/workspace/agent_runs") / job_id
plan = json.loads(base.joinpath("plan.json").read_text())
result = json.loads(base.joinpath("result.json").read_text())
print("planned_duration_sec:", plan["target_duration_sec"])
print("planned_num_frames:", plan["steps"][-1]["params"]["num_frames"])
print("actual_video_duration_sec:", result["actual_video_duration_sec"])
print("actual_final_duration_sec:", result["actual_final_duration_sec"])
print("assembly_delta_sec:", result["metadata"]["assembly"]["video_minus_planned_sec"])
PY
```

### Scene-Plan pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase2a-multiscene-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
scene_plan = json.loads(base.joinpath("scene_plan.json").read_text())
print("scene_count:", scene_plan["scene_count"])
for scene in scene_plan["scenes"]:
    print(scene["scene_id"], scene["target_duration_sec"], scene["prompt_text"])
PY
```

### Take-Plan und Auswahl pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase2b-multitake-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
state = json.loads(base.joinpath("state.json").read_text())
print("takes_per_scene:", takes["takes_per_scene"])
print("selected_take_ids:", [scene["selected_take_id"] for scene in takes["scene_outputs"]])
print("selection_mode:", state["steps"]["video"]["details"]["selection_mode"])
PY
```

### Realen End-to-End-Lauf fahren
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-run",
    "idea": "A clean cinematic startup sequence in a GPU pod.",
    "script": "System online. Render begins.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": True,
    "voice_id": "Ryan",
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Very short narration and grounded visuals.",
    "pipeline_preference": "auto",
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
```

### Final-Mux-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-mux-run",
    "idea": "A clean cinematic startup sequence in a GPU pod.",
    "script": "System online. Render begins.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": True,
    "voice_id": "Ryan",
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Very short narration and grounded visuals.",
    "pipeline_preference": "auto",
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-mux-run/final.mp4
```

### No-Voice-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-no-voice-run",
    "idea": "A compact cinematic startup scene in a GPU pod.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
```

### Multi-Segment-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-multiscene-run",
    "idea": "A compact cinematic startup scene in a GPU pod.",
    "script": "Scene one opens on the GPU pod. Scene two shows the render finishing cleanly.",
    "duration_sec": 6,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {"scene_count": 2},
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-multiscene-run/final.mp4
```

### Multi-Take-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-multitake-run",
    "idea": "A compact cinematic GPU-pod teaser.",
    "script": "Scene one shows the pod waking up. Scene two shows the render completing cleanly.",
    "duration_sec": 6,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {"scene_count": 2, "takes_per_scene": 2},
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-multitake-run/final.mp4
```

### Agent-Core Struktur pruefen
```bash
find /workspace/agent_core -maxdepth 3 -type f | sort
sed -n '1,240p' /workspace/agent_core/agent.py
sed -n '1,240p' /workspace/agent_core/planner.py
sed -n '1,260p' /workspace/agent_core/schemas.py
```

### Handoff pruefen
```bash
sed -n '1,260p' /workspace/codex/HANDOFF.md
sed -n '1,240p' /workspace/codex/PROJECT_STATE.md
sed -n '1,240p' /workspace/codex/TASK_BOARD.md
```

## Prompt-Bausteine

### Recon-Format
```text
Trenne die Antwort strikt in:
1. Verifizierte Fakten
2. Annahmen
3. Offene Fragen
4. Empfehlungen
Nenne nur Dinge als verifiziert, die lokal geprueft wurden.
```

### Architektur-Check
```text
Bewerte den Vorschlag nur fuer Phase 1 des Agent-Core.
Kein GUI-Vorschlag.
Keine n8n-Integration.
Keine API-Schicht.
Keine stillschweigende Zielaenderung.
Bevorzuge minimale, testbare, modulare Strukturen.
```

### Adapter-Design
```text
Entwirf einen duennen Adapter ueber vorhandene lokale Backends.
Nicht das zugrunde liegende Modell-Repo umbauen.
Beschreibe:
- Eingabeschema
- Ausgabeschema
- Fehlerfaelle
- State-Updates
- minimale Smoke-Tests
```

### Phase-1-Core-Review
```text
Pruefe den aktuellen Agent-Core nur fuer Phase 1.
Bewerte getrennt:
1. verifizierte Fakten
2. Annahmen
3. offene Punkte
4. Empfehlungen
Fokus:
- Planner-Regeln
- State-Store
- Adapter-Vertraege
- Artefakt-Layout
- Fehlerbehandlung
Kein GUI-, API- oder n8n-Ausbau.
```

### Session-Start fuer Codex
```text
Lies zuerst in /workspace/codex:
AGENTS.md, MISSION.md, USER_PREFERENCES.md, PROJECT_STATE.md, ACTIVE_PLAN.md, MEMORY.md, DECISIONS.md, TASK_BOARD.md und SYSTEM_AUDIT.md.
Arbeite danach nur auf Basis verifizierter Fakten.
Aktualisiere die Memory-Dateien nach relevanten Aenderungen.
```
