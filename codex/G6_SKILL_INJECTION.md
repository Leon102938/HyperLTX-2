# G6 Skill Injection

## Status
G6 aktiviert die G2/G3-Struktur als kreativen Steuerkontext, ohne Render, Modelle, Downloads, Runtime, Docker, `init.sh`, Backends, API, GUI oder n8n anzufassen.

## Was G6 Macht
User Idea -> Pipeline/Mode/Style -> Skill Injection Context -> Stage Contracts -> PromptBuilder/Review/DecisionLog Trace.

Neues zentrales Modul:
- `agent_core/creative_system/skill_injection.py`

Der `SkillInjectionContext` sammelt:
- `pipeline_id`, `mode_id`, `style_id`
- `required_skills`, `loaded_skills`, `missing_skills`
- `platform_skills`, `model_skills`, `stage_skills`, `review_skills`, `directing_skills`
- `prompt_policy`, `creative_constraints`, `anti_patterns`, `audit_hints`, `warnings`

Fehlende Skills brechen den Run nicht hart ab. Sie werden in `missing_skills` und `warnings` sichtbar.

## Wie Skills Aktiv Werden
`VideoAgent._attach_skill_trace()` baut jetzt den `SkillInjectionContext` aus Pipeline Definition, Mode, Style und Job-Metadata. Dieser Context wird in `plan.metadata.skill_injection_context` gespeichert und an `build_stage_role_contracts()` uebergeben.

Dadurch entstehen aktiv befuellte Contracts:
- `CreativeStrategy`: Plattform, Hook Pattern, Pacing, kreative Freiheit, Kontinuitaet, Audience Intent, Success Criteria, Anti-Goals, Skill-IDs.
- `BeatPlan`: Beats, Scene Roles, Timing Intent, Eskalationslogik, Payoff, Motivfamilien, Shot Recipes.
- `VisualDirection`: Motivfamilie, Shot Recipe, Kamera, Licht, Bewegung, Composition Rules, Object Count Policy, Human Action Policy, Avoid Risks.
- `ModelPromptPlan`: positive/negative Prompts, Z-Image Prompt, LTX Positive/Negative Trace, Backend Prompt Policy, Model Skills.
- `ReviewPlan`: technische Checks, Artefaktchecks, kreative Qualitaetskriterien, Plattformfit, Rejection Rules.

## Planner/Director
Der bestehende Planner/Director-Flow bleibt klein und kompatibel. G6 ersetzt keinen Executor und baut keinen neuen Beat-Planner. Die bestehende rule-based Planung erzeugt weiterhin Szene, Motive, Shot Recipes und Prompt-Metadaten; G6 macht diese Entscheidungen ueber SkillInjectionContext und Stage Contracts explizit und maschinenlesbar.

Morning Reset bleibt flexibel ueber Motivfamilien wie `light_reveal`, `tactile_object_detail`, `body_reset_gesture`, `breath_window_moment`, `fabric_texture`, `sunlight_surface` und `before_after_micro_change`. Die bisherigen Vorhang/Wasserglas/Fenster-Rezepte bleiben Bausteine, keine harte neue Pflichtarchitektur.

## PromptBuilder
Z-Image bleibt positive-only:
- `zimage_prompt_sent == positive_model_prompt`
- keine Avoid-Liste
- keine Debuglabels
- keine Script-Saetze

LTX bleibt positiv plus kurze Avoid Policy:
- `ltx_positive_prompt_sent`
- `ltx_negative_prompt_sent`
- `ltx_prompt_sent` als kombinierter sicherer Backend-Fallback, weil kein Adapter-Umbau in G6 gemacht wurde

`DebugPrompt` bleibt Audit-/Debug-Material und wird nicht als Backend-Prompt markiert.

## Review
ReviewPlan und Qwen3-VL-Systemprompt kennen jetzt zusaetzlich:
- boring scene
- weak hook
- unclear action
- generic stock feel
- physical incoherence
- bad composition
- poor platform fit
- no visual change
- dead/static scene
- confusing subject
- mismatch between voice/script and visuals
- low phone-size readability

Heuristiken bleiben ehrlich: `real_vlm_inference_used` wird nicht auf True gesetzt, wenn keine echte VLM-Inferenz lief.

## Artefakte Lesen
`prompt_audit.json`:
- SkillInjectionContext, Stage Contracts, Prompt Checks, Backend Policy, Scene Prompt Audit.

`model_prompts.json`:
- pro Szene positive/negative Prompts, Z-Image/LTX Trace, Policy Checks, Stage Contracts.

`stage_contracts.json`:
- kanonischer Contract-Snapshot fuer CreativeStrategy, BeatPlan, VisualDirection, ModelPromptPlan und ReviewPlan.

`decision_log.json`:
- Pipeline, Mode, Style, Skill Set, Hook Pattern, Motif Family, Shot Recipe, CreativeStrategy, BeatPlan, ReviewPlan und Gate-/Stop-Status.

## Smoke
Sicherer Smoke ohne Render:
- Run: `/workspace/agent_runs/g6-skill-injection-stop-after-model-prompts-smoke`
- Ergebnis: `prompt_audit.json`, `model_prompts.json`, `stage_contracts.json`, `decision_log.json` vorhanden; `final.mp4` nicht vorhanden.

## Future Work
- Echter Creative Strategy / Beat Planner 2.0 als eigene Planungsstufe.
- Director-LLM-Systemprompt tiefer mit SkillInjectionContext befuellen, ohne Backend-/Runtime-Umbau.
- Adapter-separates LTX-`negative_prompt` nur nach explizitem Backend-Vertrag.
- DecisionLog spaeter append-only bei echter Take Selection und Final Quality Gate fortschreiben.

## G7
Naechster sinnvoller Schritt: G7 Creative Strategy / Beat Planner 2.0.
