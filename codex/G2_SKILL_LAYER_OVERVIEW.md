# G2 Skill Layer Overview

## Zweck
G2 macht die Content Maschine skill-aware. Skills sind kleine, ladbare Markdown-Vertraege fuer Modelle, Plattformen, Produktionsstufen, Regie, Prompting und Review. Sie ersetzen noch nicht den Executor, aber sie machen sichtbar, welches Produktionswissen eine Pipeline erwartet und welche Regeln spaeter Director, Planner, Prompt Builder, Reviewer und Selector beeinflussen sollen.

## Skill-Typen
- `models/`: model-specific Regeln fuer Z-Image, LTX, Qwen3-VL Review und Qwen TTS.
- `platforms/`: Plattformkontext fuer TikTok Shortform, Instagram Reels und YouTube Shorts.
- `stages/`: Produktionsstufen wie Creative Strategy, Beat Planning, Visual Direction, Model Prompting, Quality Review und Take Selection.
- `directing/`: Regie-Faehigkeiten fuer Shortform, Clean Lifestyle, Motion/Camera und Anti-Boring.
- `prompting/`: Prompt-spezifische Regeln fuer positive Image Prompts, Video Prompts und Negative Prompt Policy.
- `review/`: Review-Faehigkeiten fuer visuelle Qualitaet, Postability, langweilige Szenen und Artefakte.

Jede Skill-Datei hat dieselben Pflichtfelder: `title`, `purpose`, `when_to_use`, `rules`, `do`, `dont`, `output_contract`, `common_failures`, `audit_hints`.

## Laden und Trace
Der Loader liegt in `agent_core/creative_system/skill_loader.py`.

Wichtige Funktionen:
- `load_skill(path_or_id)`: laedt eine Markdown-Skill-Datei per ID wie `models/zimage_turbo` oder per Pfad.
- `load_required_skills(skill_ids)`: laedt mehrere Skills, dedupliziert IDs und meldet fehlende Skills als `missing`.
- `resolve_skills_for_pipeline(pipeline_def, mode, style)`: kombiniert Skills aus Pipeline Definition, Pipeline Steps, Mode und Style.

Der Agent schreibt Skill-Trace in `plan.metadata`, `prompt_audit.json` und `model_prompts.json`:
- `required_skills`
- `loaded_skills`
- `missing_skills`
- `stage_roles`

## Pipeline Definitions
Pipeline Definitions koennen jetzt `required_skills` auf Pipeline- und Step-Ebene deklarieren. `stage_roles` beschreibt fuer Menschen und spaetere CLI/API-Ansichten, welche Rolle eine Stage hat.

Neue Pipeline:
- `agent_core/pipeline_defs/clean_shortform_v1.json`

Ziel:
- kurze Social-Videos
- flexible Creative Strategy
- Beat Plan
- Visual Direction
- Model Prompt Compile
- Review
- Selection
- Assembly
- Final Quality Gate

`simple_video_v1` bleibt erhalten und rueckwaertskompatibel.

## Creative Roles
G2 fuehrt Datenvertraege ein, ohne den Executor neu zu schreiben:
- `CreativeStrategy`
- `BeatPlan`
- `VisualDirection`
- `ModelPromptPlan`
- `ReviewPlan`

Diese Schemas trennen kuenftig klarer:
- User Intent und Strategie
- Beat-/Motivplanung
- visuelle Regie
- backend-spezifische Prompts
- Review- und Selektionskriterien

## Morning Reset
Morning Reset ist nicht mehr nur eine starre Vorhang/Wasserglas/Fenster-Sequenz. Der Mode enthaelt jetzt flexible `motif_families`:
- `light_reveal`
- `tactile_object_detail`
- `body_reset_gesture`
- `breath_window_moment`
- `fabric_texture`
- `sunlight_surface`
- `before_after_micro_change`

Die bestehenden Shot Recipes bleiben als Bausteine erhalten, aber sie sind nicht mehr als einzige Pflichtszenen zu verstehen.

## Model-Specific Policy
Z-Image:
- positive-only
- kurze visuelle Prompts
- keine Avoid-Liste
- keine Debuglabels
- keine Script-Saetze

LTX:
- positiver Video-Prompt
- kurze negative/avoid Liste
- G2 dokumentiert `ltx_positive_prompt_sent` und `ltx_negative_prompt_sent`
- separater Adapter-`negative_prompt` Support ist noch nicht verdrahtet

Qwen3-VL Review:
- soll Artefakte, langweilige Szenen, weak hook, unclear action, generic stock feel, visual incoherence, composition und platform-fit weakness pruefen

## Decision Log
`decision_log.json` ist als Run-Artefakt vorbereitet. Es protokolliert initial:
- `selected_pipeline`
- `selected_mode`
- `selected_style`
- `selected_skill_set`
- `selected_motif_family`
- `selected_shot_recipe`
- `backend_prompt_policy`
- Platzhalter fuer `selected_take`
- Platzhalter fuer `final_quality_decision`

Vollstaendige Laufzeit-Entscheidungen nach Take Selection und Final Quality Gate bleiben Folgearbeit.

## Neue Modes, Styles, Pipelines
Neue Skills:
1. Markdown-Datei unter passender Skill-Kategorie anlegen.
2. Pflichtfelder ausfuellen.
3. Skill-ID in Pipeline Definition, Mode oder Style als `required_skills` referenzieren.
4. Tests ergaenzen, die `loaded_skills` und `missing_skills` pruefen.

Neue Pipeline:
1. JSON unter `agent_core/pipeline_defs/<pipeline_id>.json` anlegen.
2. `required_skills`, `stage_roles`, Steps, Checkpoints und Policies definieren.
3. Bestehende Executor-Schrittnamen nur dann verwenden, wenn sie wirklich schon ausgefuehrt werden.
4. Neue Stages duerfen vorbereitet werden, aber brauchen spaeter einen Executor-Schritt.

## Noch Nicht Umgesetzt
- Skills steuern den Director/Planner noch nicht aktiv als Prompt-Kontext; G3 macht dafuer jetzt Stage Contracts tracebar.
- Es gibt keinen Provider/Tool Selector.
- Es gibt noch keinen Resume-Executor fuer blockierte Approval Gates.
- LTX separater `negative_prompt` Adapter-Support ist nicht implementiert.
- Decision Log schreibt noch keine echten Take-Selection- und Final-Verdict-Entscheidungen nach Abschluss, enthaelt aber ab G5 `approval_gate_status`, `stop_after` und `quality_decision`-Contract.
- Keine n8n/API/GUI-Integration in G2.
