# G3 Stage Roles

## Status
G3 ist defensiv umgesetzt. Es gibt klare Stage Role Contracts, aber keinen neuen Executor.

## Artefakt
Pro geplantem Run wird `stage_contracts.json` geschrieben. Dieselben Daten erscheinen auch in:
- `plan.json` unter `metadata.stage_contracts`
- `prompt_audit.json`
- `model_prompts.json`

## Contracts
- `CreativeStrategy`: Mode, Style, Plattform, Hook, Pacing, kreative Freiheit, Kontinuitaet, Zielkriterien und Anti-Goals.
- `BeatPlan`: Beats, Scene Roles, Timing Intent, Eskalationslogik, Payoff, Motive und Shot Recipes.
- `VisualDirection`: Motivfamilie, Shot Recipe, Kamera, Licht, Bewegung, Composition Rules, Object Count Policy, Human Action Policy und Avoid Risks.
- `ModelPromptPlan`: Positive/negative Prompts, Z-Image Prompt, LTX Positive/Negative Trace, Backend Prompt Policy und geladene Model Skills.
- `ReviewPlan`: technische Checks, Artifact Checks, Creative Quality Checks, Platform Fit Checks und Rejection Rules.

## Nutzung
Heute sind die Contracts Diagnose- und Anschluss-Artefakte. Morgen koennen Director, Planner und Prompt Builder diese Contracts aktiv als Eingabe nutzen.

## Future Work
- Skills direkt in Director-Systemprompt und Planner-Auswahl einspeisen.
- BeatPlan wirklich als separate Planungsstufe ausfuehren.
- VisualDirection pro Szene statt nur global/erstem Szenenkontext staerker ausbauen.
- ReviewPlan in Qwen-/Heuristik-Reviewer als explizites JSON-Contract-Feld durchreichen.
