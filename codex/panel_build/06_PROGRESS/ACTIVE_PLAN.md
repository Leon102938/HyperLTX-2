# Active Plan

Stand: 2026-05-15, Skill Tree V1 in Phase 1 Stage 03-08.

## Aktueller Build-Stand

Textual bleibt fest auf der stabilen 0.89.x-Linie. Im Pod ist `textual==0.89.1` aktiv, und der Pin `textual>=0.89,<1.0` bleibt Pflicht. Keine neuen Dependency-Experimente und keine Textual-8.x-Anpassungen.

Phase 1 baut jetzt eine lokale CLI-Runtime fuer `creative-os run-phase1`, die strukturierte Creative-OS-Artefakte unter `/workspace/agent_runs/<job-id>/creative_os/` schreibt. Der Fokus liegt auf Runtime/Integration bis Stage `09`, nicht auf weiterem Design-Feinschliff.

Der zusaetzliche Live-Pfad `creative-os run-phase1-live` schreibt waehrend des Runs `live_status.json` und `stage_events.jsonl`. `viewed_stage` bleibt Cockpit-Auswahl, `real_run_stage/current_running_stage` bleibt Runner-Wahrheit.

V2-Korrektur: Stage `09` wird nur `done`, wenn `keyframe_manifest.json` alle Jobs als `finished` mit `file_exists=true` meldet. Disabled/missing Backend bleibt `paused_missing_backend`, completed nur `00` bis `08`, Stage `09=error`.

V3-Korrektur: Stage `10` bis `15` sind fuer Phase-1-Runs out-of-scope/not built und koennen in der Pipeline Map nicht gruen werden. Stage `09` zeigt echte Manifest-/Live-Daten im Workspace und schreibt das Manifest waehrend echter Image-Generierung fort. Watch-Refresh bleibt semantisch und ueberschreibt manuelle Stage-Auswahl nicht.

Skill Tree V1 ist jetzt der echte Stage-03-Input fuer Phase 1. Der Loader liest `/workspace/skills/skill_manifest.json`, laedt Mode-, Style-, Hook- und Model-Skill-Dateien und schreibt `skill_match.json` sowie `skill_tree.json`. Fehlende Skill-Dateien werden als `missing`/`unknown` gemeldet, nicht gefaked.

Stage `04` bis `08` lesen die geladenen Regeln minimal, aber echt: `creative_strategy.json` bekommt Mode/Style-Regeln, `beat_hook_plan.json` Hook-Regeln, `creative_judge.json` aktive Skill-Regeln, `scene_contracts.json` Style/Model-Regeln und `prompt_payload_compiled.json`/`zimage_prompts.json` die Z-Image-Textartefaktregeln.

CLI:

```bash
python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1 \
  --job-id creative-os-jungle-001 \
  --topic "jungle safari at sunrise" \
  --pipeline shortform_storyboard_v1 \
  --mode visual_adventure \
  --style cinematic_nature \
  --format portrait \
  --duration 9s \
  --scenes 3
```

Retry/Resume nur fuer Stage `09`:

```bash
python3 /workspace/scripts/agent_core_cli.py creative-os retry-keyframes \
  --job-id phase1-hardening-smoke-20260513 \
  --runs-root /workspace/agent_runs
```

Optionen: `--dry-run`, `--scene scene_02`, `--force`. Ohne `--force` werden fertige vorhandene PNGs nicht neu erzeugt.

Live-Run plus Watch:

```bash
python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1-live \
  --job-id live-smoke-20260514 \
  --topic "jungle safari at sunrise" \
  --pipeline shortform_storyboard_v1 \
  --mode visual_adventure \
  --style cinematic_nature \
  --format portrait \
  --duration 9s \
  --scenes 3 \
  --no-generate-images

python3 /workspace/scripts/creative_os_cockpit.py \
  --job-id live-smoke-20260514 \
  --runs-root /workspace/agent_runs \
  --watch \
  --refresh-sec 1
```

Echte Z-Image-Generierung:

```bash
python3 /workspace/scripts/agent_core_cli.py creative-os run-phase1-live \
  --job-id live-v2-smoke-images-20260514 \
  --topic "jungle safari at sunrise" \
  --pipeline shortform_storyboard_v1 \
  --mode visual_adventure \
  --style cinematic_nature \
  --format portrait \
  --duration 9s \
  --scenes 3 \
  --generate-images
```

`--open-cockpit` startet Textual nicht im selben TTY-Background. Der Flag gibt sichere Terminal-Befehle aus. Fuer sichtbare schnelle Stages: `--stage-delay-seconds 0.5`.

## Phase-1-Artefakte 00-09

- Stage `00 Command Center`: `normalized_job.json`
- Stage `01 Pipeline wählen`: `pipeline_route.json`, kompatibel dazu `intent_route.json`
- Stage `02 Mode & Style`: `mode_style.json`, kompatibel dazu `creative_direction.json`
- Stage `03 Skills laden`: `skill_match.json`, `skill_tree.json`
- Stage `04 Creative Strategy`: `creative_strategy.json`
- Stage `05 Beat / Hook Planner`: `beat_hook_plan.json`, kompatibel dazu `selected_beat_plan.json`
- Stage `06 Creative Judge`: `creative_judge.json`, kompatibel dazu `stage6_review_decision.json`
- Stage `07 Scene Contracts`: `scene_contracts.json`, kompatibel dazu `keyframe_contracts.json`
- Stage `08 Prompt Compiler`: `prompt_payload_compiled.json`, `zimage_prompts.json`
- Stage `09 Image / Keyframe Generation`: `keyframe_manifest.json`, `phase1_status.json`
- Live-State: `live_status.json`, `stage_events.jsonl`

Stage `09` prueft das vorhandene lokale Z-Image-HTTP-Backend. Wenn es erreichbar ist, werden Jobs ueber den bestehenden `/zimage/jobs`-Pfad versucht. Wenn es fehlt, bleibt der Run sauber auf `phase1_paused_missing_image_backend`; das Manifest zeigt pro Szene `status=error` und es werden keine Fake-Bilder als Erfolg behauptet.

Reality-Fix: `phase1_status.json` unterscheidet jetzt `real_run_stage`, `last_completed_stage` und `next_available_stage`. Bei abgeschlossenem Stage `09` enthaelt `completed_stages` auch `09`; `next_available_stage=none_phase1_complete` bedeutet ausdruecklich `Stage 10+ not built yet`.

Hardening: `retry-keyframes` liest ausschliesslich `keyframe_manifest.json`, erkennt `failed/error/queued/running`, fehlende `output_path` und `file_exists=false`, fuehrt nur diese Jobs erneut aus und schreibt nur `keyframe_manifest.json`, `phase1_status.json` sowie ggf. `keyframe_gallery.html` neu. Stage `00` bis `08` werden nicht neu geschrieben.

## Weiter bestehender Panel-Stand

Stage `00` bis `09` bleiben als V1-Basis im Active Workspace erhalten:

- klare geschlossene Panels
- bessere Platznutzung
- mehr innere Struktur
- keine ASCII-Pipe-Trenner
- keine offenen Rahmen
- lange Texte werden gekuerzt
- Stage Map bleibt konsistent zur gewaehlten Stage

## Stage 00-09 Status

- Stage `00 Command Center`: stabiler Kontroll-/Statusbereich beibehalten; keine CLI-Eingabe gebaut.
- Stage `01 Pipeline wählen`: Current-Position-Block entfernt; Pipeline Purpose/Overview, Pipeline Flow, Pipeline Assets und Output/Next sichtbar.
- Stage `02 Mode & Style`: bestehende Struktur erhalten und mit Mode Intent, Style Language, Visual Rules, Risks/Avoids und Handoff ergaenzt.
- Stage `03 Skills laden`: Skill Tree V1 als sichtbarer Tree, Skill Loading Progress, Loading Status, Skill Sources und Health; Pipeline selbst wird nicht als Skill behandelt.
- Stage `04 Creative Strategy`: A/B/C/D-Struktur mit Strategy Engine, Input Context, Skill Stack, JSON Preview und Output Summary.
- Stage `05 Beat / Hook Planner`: Hook Brief, textbasierte Hook Candidates, Selected Beat Plan und Output Preview; keine Fake-Bilder.
- Stage `06 Creative Judge`: Judge Input, Creative Checks, Final Creative Decision, Output Preview, Risiken/Fixes/Handoff.
- Stage `07 Scene Contracts`: A/B/C-Struktur erhalten; pro Szene Status, Visual Anchor, Environment, Action, Camera/Lighting, Allowed/Forbidden Visuals und Risk sichtbarer; Readiness verbessert.
- Stage `08 Prompt Compiler`: stabiler geschlossener `IMAGE COMPILER (ACTIVE)` Hauptkasten bleibt erhalten; rechte Video/Audio/Music-Compiler-Spalte bleibt erhalten.

## Nicht Geaendert

- Stage `09` liest jetzt echte `keyframe_manifest.json` Jobs, inklusive Backend, Backend-Status, Overall-Status, Progress, Elapsed, Output-Pfad, Error, Backend-Job-ID und Datei-Metadaten.
- Stage `09` zeigt echte Preview-Pfade statt Fake-Thumbnails. Fertige Jobs mit fehlender Datei werden als Error/Warn angezeigt.
- Neue Runs erzeugen bei vorhandenen PNGs eine `keyframe_gallery.html`.
- Fehlt `keyframe_manifest.json` in einem Real-Run, zeigt Stage `09` `missing manifest` und keine Fake-Cards.
- Header, Sidebar, System Status und Bottom Panels nicht redesignt.
- Active-Workspace-Scroll bleibt.
- Keine Stage-10-bis-15-Runtime, keine LTX-Video-Generation, kein Assembly, kein n8n/API.
- Keine Scroll-, Terminal-, Quit- oder Performance-Fixes angefasst.
- Keine Flow-/Symbol-Leiste im Active Workspace zurueckgebracht.

## Teststatus

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 16 Tests.
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 21 Tests.
- Hardening-Smoke `/workspace/agent_runs/phase1-hardening-smoke-20260513/creative_os`: `phase1_finished_stage09`, 3 echte PNGs, `keyframe_gallery.html`.
- Retry-Smoke `/workspace/agent_runs/phase1-hardening-retry-sim-20260513/creative_os`: fehlende `scene_02.png` per `retry-keyframes --scene scene_02` neu erzeugt.

## Naechster enger Build-Schritt

Stage-09-Review/Gate definieren, das echte Keyframes bewertet, ohne Stage `10+` Runtime zu bauen.
