# G7 Creative Beat Planner

## Zweck
G7 macht aus dem G6-Skill- und Contract-Wiring eine aktive kreative Entscheidungsstufe. `clean_shortform_v1` plant nicht mehr automatisch eine starre Morning-Reset-Folge wie Vorhang, Glas, Fenster, sondern erzeugt mehrere plausible Beat-Varianten, bewertet sie und nutzt den besten Candidate fuer Szene, VisualDirection und ModelPromptPlan.

## CreativeIntent
`agent_core/creative_system/strategy_planner.py` analysiert User-Idee und Script ohne LLM oder Modellstart. Der Intent enthaelt unter anderem:
- `raw_user_idea`
- `raw_script`
- `sanitized_visual_intent`
- `topic`
- `platform`
- `desired_emotion`
- `content_promise`
- `audience_value`
- `visual_energy`
- `pacing_type`
- `risk_profile`
- `constraints`
- `inferred_mode`
- `inferred_style`

`sanitized_visual_intent` ist bewusst keine Kopie des Scripts. Imperative wie "Open soft light", "Place one clear glass" oder "Breathe by the window" werden in semantische visuelle Absicht ueberfuehrt.

## BeatPlanCandidates
G7 erzeugt fuer Clean Shortform/Morning Reset mindestens drei Kandidaten:
- `light_to_action`: sichtbarer Lichtwechsel, taktiles Detail, ruhiger menschlicher Payoff.
- `tactile_first`: Objekt-/Textur-Hook, Body-Reset-Geste, Licht-Payoff.
- `motion_first`: menschliche Aktion sofort, Environment Response, ruhiger Close.

Jeder Candidate enthaelt Hook Pattern, Beat Sequence, Scene Roles, Motif Families, Shot Recipes, Continuity Strategy, Platform Fit Intent, erwarteten visuellen Wandel, Risk Notes, Rationale und Score Placeholder.

## Scoring
`score_beat_plan_candidate` bewertet Hook, visuelle Klarheit, Action-Lesbarkeit, Originalitaet, Modellmachbarkeit, Artefaktrisiko, Platform Fit, Continuity und Anti-Boring. Statische, generische oder text-/UI-nahe Kandidaten verlieren Punkte. Klare physische Aktion, sichtbarer Wandel, einfacher Bildaufbau und starker erster Hook gewinnen Punkte.

## Planner Integration
Der Planner aktiviert G7 fuer `clean_shortform_v1` oder explizites `enable_g7_beat_planner`. Legacy-Pfade bleiben unveraendert. Fuer aktive G7-Runs speichert der Planner:
- `creative_intent`
- `beat_plan_candidates`
- `beat_plan_candidate_scores`
- `selected_beat_plan_candidate`
- `selected_candidate_id`
- `selected_hook_pattern`
- `selected_motif_sequence`
- `selected_shot_recipe_sequence`
- `per_scene_visual_direction`

Die ausgewaehlte Candidate-Struktur wird in `SceneIntent` uebernommen, bevor der PromptBuilder Szenenprompts baut.

## PromptBuilder Integration
Der PromptBuilder liest per-scene VisualDirection aus `director_output.metadata`. Pro Szene werden Motif, Shot Recipe, Action, Camera, Lighting, erlaubte Visuals und Avoid Risks in den Scene World Contract uebernommen. Z-Image bleibt positive-only; LTX bekommt getrennte positive und kurze negative Trace-Felder.

## Artefakte Lesen
- `stage_contracts.json`: Intent, Candidates, Scores, selected Candidate, BeatPlan und per-scene VisualDirection.
- `prompt_audit.json`: Skill/Contract-Kontext plus selected Candidate und per-scene Direction pro Szene.
- `model_prompts.json`: model-facing Prompts, Z-Image/LTX Trace und G7 Candidate-Kontext.
- `decision_log.json`: Candidate-Auswahl mit Score-Breakdown, Motif-/Shot-Sequenz und Selection Reason.

## Smoke
Sicherer Dry-Run:
- Run: `/workspace/agent_runs/g7-beat-planner-stop-after-model-prompts-smoke`
- Stop: `model_prompts`
- Kein `final.mp4`
- Keine Render, Modelle, Downloads oder Backend-Umbauten.

## Future Work
- G8 Feedback Loop / Retry Executor: Review-Warnings kontrolliert in Replan-, Regenerate- oder Stop-Entscheidungen ueberfuehren.
- Spaeterer kontrollierter erster Render-Test nur mit explizitem Render-Auftrag und engen Stop-/Review-Gates.
