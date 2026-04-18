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
curl -sS http://127.0.0.1:8000/agent-core/jobs/does-not-exist
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

### Phase-4A-Bridge lokal pruefen
```bash
curl -sS -X POST http://127.0.0.1:8000/agent-core/run \
  -H 'Content-Type: application/json' \
  --data @/workspace/examples/agent_core_bridge_request.json

curl -sS http://127.0.0.1:8000/agent-core/jobs/bridge-demo-job
```

### Phase-4B-Bridge lokal pruefen
```bash
curl -sS -X POST http://127.0.0.1:8000/agent-core/jobs \
  -H 'Content-Type: application/json' \
  --data @/workspace/examples/agent_core_bridge_request.json

curl -sS http://127.0.0.1:8000/agent-core/jobs/<job_id>
```

### Phase-4C-Response-Haertung lokal pruefen
```bash
curl -sS http://127.0.0.1:8000/agent-core/jobs/<job_id> | python - <<'PY'
import json, sys
payload = json.load(sys.stdin)
print("status:", payload["status"])
print("status_summary:", payload["status_summary"])
print("is_terminal:", payload["is_terminal"])
print("should_poll:", payload["should_poll"])
print("retry_after_sec:", payload["retry_after_sec"])
print("artifacts_ready:", payload["artifacts_ready"])
print("result_json_ready:", payload["result_json_ready"])
print("final_mp4_ready:", payload["final_mp4_ready"])
print("public_result_json_url:", payload["public_refs"]["result_json_url"])
print("public_final_mp4_url:", payload["public_refs"]["final_mp4_url"])
PY
```

### Phase-4B-Bridge ueber Proxy pruefen
```bash
curl -sS -X POST https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/jobs \
  -H 'Content-Type: application/json' \
  --data @/workspace/examples/agent_core_bridge_request.json

curl -sS https://mvwg65x59mc01e-8000.proxy.runpod.net/agent-core/jobs/<job_id>
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
job_id = "real-phase2c-quality-guard-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
state = json.loads(base.joinpath("state.json").read_text())
print("takes_per_scene:", takes["takes_per_scene"])
print("selected_take_ids:", [scene["selected_take_id"] for scene in takes["scene_outputs"]])
print("selection_mode:", state["steps"]["video"]["details"]["selection_mode"])
print("total_retry_count:", takes["total_retry_count"])
for scene in takes["scene_outputs"]:
    print(
        scene["scene_id"],
        "selected=",
        scene["selected_take"]["review_status"],
        "valid=",
        scene["selected_take"]["validation"]["validation_status"],
        "retries=",
        len(scene["retry_history"]),
    )
PY
```

### Quality-Guard im Detail pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase2c-quality-guard-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
for scene in takes["scene_outputs"]:
    print("scene:", scene["scene_id"])
    for take in scene["takes"]:
        validation = take.get("validation") or {}
        print(
            " ",
            take["take_id"],
            take["review_status"],
            validation.get("validation_status"),
            validation.get("width"),
            validation.get("height"),
            validation.get("fps"),
            validation.get("duration_sec"),
            validation.get("duration_delta_sec"),
            validation.get("issues"),
        )
PY
```

### Varianten im Detail pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase2d-variation-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
scene = takes["scene_outputs"][0]
print("variations_per_scene:", takes["variations_per_scene"])
print("takes_per_variation:", takes["takes_per_variation"])
print("selected_take_id:", scene["selected_take_id"])
print("selected_variation_id:", scene["selected_variation_id"])
for variation in scene["variations"]:
    print(
        variation["variation_id"],
        variation["shot_type"],
        variation.get("camera_style"),
        variation.get("camera_motion"),
        variation["framing_hint"],
    )
for take in scene["takes"]:
    print(
        take["take_id"],
        take["variation_id"],
        take["review_status"],
        take["validation"]["validation_status"],
    )
PY
```

### Kreative Auswahl im Detail pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase2e-creative-selection-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
state = json.loads(base.joinpath("state.json").read_text())
print("selection_mode:", takes["selection_mode"])
print("creative_selection_mode:", takes["creative_selection_mode"])
for scene in takes["scene_outputs"]:
    print(
        scene["scene_id"],
        scene["selected_take_id"],
        scene["selected_take"]["shot_type"],
        scene["technical_score"],
        scene["creative_score"],
        scene["selected_by_rule"],
        scene["selection_reason"],
    )
    for candidate in scene["selection"]["scored_candidates"]:
        print(
            " ",
            candidate["take_id"],
            candidate["shot_type"],
            candidate["technical_score"],
            candidate["creative_score"],
            candidate["selected_by_rule"],
        )
print("state-selected:", state["steps"]["video"]["details"]["selected_scene_outputs"])
PY
```

### Storyboard-/Keyframe-Details pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase3a-storyboard-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
story = json.loads(base.joinpath("storyboard_plan.json").read_text())
takes = json.loads(base.joinpath("takes.json").read_text())
state = json.loads(base.joinpath("state.json").read_text())
print("storyboard_backend:", state["steps"]["storyboard"]["backend_name"])
print("selection_mode:", story["selection_mode"])
for scene in story["scene_storyboards"]:
    print("scene:", scene["scene_id"])
    print(" selected_keyframe:", scene["selected_keyframe"]["candidate_id"] if scene["selected_keyframe"] else None)
    print(" selected_variation:", scene["selected_keyframe"]["variation_id"] if scene["selected_keyframe"] else None)
    for candidate in scene["generated_candidates"]:
        validation = candidate.get("validation") or {}
        print(
            " ",
            candidate["candidate_id"],
            candidate["review_status"],
            validation.get("validation_status"),
            validation.get("width"),
            validation.get("height"),
        )
print("take-keyframe-relation:", takes["scene_outputs"][0]["selected_take"]["metadata"]["selected_keyframe"])
PY
```

### Phase-3B-Rendermodus pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "real-phase3b-keyframe-1"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
result = json.loads(base.joinpath("result.json").read_text())
scene = takes["scene_outputs"][0]
print("video_mode:", scene["video_mode"])
print("render_mode:", scene["render_mode"])
print("fallback_reason:", scene["fallback_reason"])
print("selected_keyframe_usage:", scene["selected_keyframe_usage"])
print("render_mode_counts:", result["metadata"]["render_mode_counts"])
PY
```

### Phase-4A-Bridge-Artefakte pruefen
```bash
python - <<'PY'
import json, pathlib
job_id = "bridge-demo-job"
base = pathlib.Path("/workspace/agent_runs") / job_id
result = json.loads(base.joinpath("result.json").read_text())
print("success:", result["success"])
print("output_final_path:", result["output_final_path"])
print("result_json:", base / "result.json")
print("state_json:", base / "state.json")
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

### Phase-2C-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-phase2c-run",
    "idea": "A compact cinematic GPU-pod teaser.",
    "script": "Scene one shows the pod waking up. Scene two shows the render completing cleanly.",
    "duration_sec": 6,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {"scene_count": 2, "takes_per_scene": 2, "max_take_retries_per_scene": 1},
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
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-phase2c-run/final.mp4
```

### Phase-3B-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-phase3b-run",
    "idea": "A cinematic GPU pod render starts from a selected storyboard keyframe.",
    "script": "The pod wakes up and the render interface comes alive.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "use_storyboard": True,
    "video_mode": "keyframe_conditioned",
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals, clean composition, readable first frame.",
    "pipeline_preference": "auto",
    "metadata": {
        "force_single_scene": True,
        "variations_per_scene": 2,
        "takes_per_scene": 1,
        "storyboard_candidates_per_scene": 2
    },
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        },
        "zimage": {
            "steps": 9,
            "guidance_scale": 0.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
python - <<'PY'
import json, pathlib
job_id = "manual-real-phase3b-run"
base = pathlib.Path("/workspace/agent_runs") / job_id
takes = json.loads(base.joinpath("takes.json").read_text())
scene = takes["scene_outputs"][0]
print("render_mode:", scene["render_mode"])
print("selected_keyframe_usage:", scene["selected_keyframe_usage"])
PY
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-phase3b-run/final.mp4
```

### Phase-4A-End-to-End-Lauf ueber HTTP
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

```bash
curl -sS -X POST http://127.0.0.1:8010/agent-core/run \
  -H 'Content-Type: application/json' \
  --data @/workspace/examples/agent_core_bridge_request.json

curl -sS http://127.0.0.1:8010/agent-core/jobs/bridge-demo-job
```

### Phase-2D-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-phase2d-run",
    "idea": "A compact cinematic GPU-pod teaser.",
    "script": "One scene shows the pod waking up and settling into a clean hero state.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {"force_single_scene": True, "variations_per_scene": 2, "takes_per_scene": 1, "max_take_retries_per_scene": 1},
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
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-phase2d-run/final.mp4
```

### Phase-2E-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-phase2e-run",
    "idea": "A compact cinematic GPU-pod teaser.",
    "script": "Scene one shows the pod waking up. Scene two shows render progress moving across the interface before a clean resolve.",
    "duration_sec": 6,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {"scene_count": 2, "variations_per_scene": 2, "takes_per_scene": 1, "max_take_retries_per_scene": 1},
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
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-phase2e-run/final.mp4
```

### Phase-3A-End-to-End-Lauf
```bash
python - <<'PY'
from agent_core.agent import VideoAgent
job = {
    "job_id": "manual-real-phase3a-run",
    "idea": "A compact cinematic GPU-pod storyboard check.",
    "script": "One scene shows the pod waking up and settling into a clear readable frame.",
    "duration_sec": 4,
    "orientation": "landscape",
    "resolution": "768x448",
    "use_voice": False,
    "use_storyboard": True,
    "style": "cinematic tech trailer",
    "extra_llm_instruction": "Grounded visuals only.",
    "pipeline_preference": "auto",
    "metadata": {
        "force_single_scene": True,
        "variations_per_scene": 2,
        "takes_per_scene": 1,
        "storyboard_candidates_per_scene": 2,
        "max_take_retries_per_scene": 1
    },
    "backend_overrides": {
        "ltx2": {
            "num_inference_steps": 8,
            "video_cfg_guidance_scale": 3.0,
            "audio_cfg_guidance_scale": 3.0
        },
        "zimage": {
            "steps": 6,
            "guidance_scale": 0.0
        }
    }
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.model_dump())
PY
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration -of json /workspace/agent_runs/manual-real-phase3a-run/final.mp4
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

### Phase-5A Director-Check
```bash
python - <<'PY'
from agent_core import VideoAgent
job = {
    "job_id": "manual-phase5a-director-check",
    "idea": "A director layer should sharpen the opening and style lock.",
    "script": "The system wakes, frames the workspace, and resolves cleanly.",
    "duration_sec": 6,
    "use_voice": False,
    "resolution": "320x256",
    "orientation": "landscape",
    "metadata": {"force_single_scene": True, "variations_per_scene": 2}
}
result = VideoAgent().run_job(job, raise_on_error=False)
print(result.metadata.get("director_mode"))
print(result.metadata.get("director_fallback_reason"))
PY
sed -n '1,220p' /workspace/agent_runs/manual-phase5a-director-check/director_output.json
```

### Phase-5B Director-Modell holen
```bash
/workspace/scripts/download_director_model.py
```

### Phase-5B Director-Serve starten
```bash
DIRECTOR_LLM_N_GPU_LAYERS=8 \
DIRECTOR_LLM_CTX=2048 \
DIRECTOR_LLM_REASONING=off \
DIRECTOR_LLM_FLASH_ATTN=on \
/workspace/scripts/serve_director_llm.sh
```

### Phase-5B Director-Serve pruefen
```bash
DIRECTOR_LLM_BASE_URL=http://127.0.0.1:8011 \
DIRECTOR_LLM_MODEL=Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
/workspace/scripts/check_director_llm.py

curl -sS http://127.0.0.1:8011/v1/models
```

### Lokalen Director-LLM pruefen
```bash
python - <<'PY'
from agent_core.planner import ProductionPlanner
from agent_core.backend_registry import build_default_registry
from agent_core.schemas import JobInput
planner = ProductionPlanner(build_default_registry())
job = JobInput(
    idea="A configured local director endpoint should be used.",
    script="The system wakes into frame.",
    use_voice=False,
    resolution="draft",
    orientation="landscape",
    metadata={
        "director_llm": {
            "profile": "qwen36_llama_cpp_local",
            "base_url": "http://127.0.0.1:8011",
            "model": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
        }
    }
)
plan = planner.build_plan(job)
print(plan.director_output.mode)
print(plan.director_output.llm_model)
print(plan.director_output.fallback_reason)
PY
```
