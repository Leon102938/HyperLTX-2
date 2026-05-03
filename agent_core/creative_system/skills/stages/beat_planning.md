## title
Beat Planning Stage Skill

## purpose
Turn creative strategy into a small set of visual beats with role, action, and progression.

## when_to_use
Use when building scene_plan.json and variation plans.

## rules
- Each beat needs one visible action or change.
- Beats should differ in scale or function.
- Recipes are selectable building blocks, not mandatory fixed scenes.

## do
- Assign hook, development, and payoff roles.
- Select motif family and candidate shot recipe.
- Keep scene count aligned with duration.

## dont
- Do not repeat the same static setup.
- Do not force scenes unrelated to the user idea.
- Do not depend on text overlays for meaning.

## output_contract
- BeatPlan records beats, selected_motif_families, selected_shot_recipes, and transition notes.

## common_failures
- Clip is boring despite clean prompts.
- Voice fits but visuals do not.
- All scenes feel the same.

## audit_hints
- Inspect scene_plan.json scene_intents and prompt_build_metadata motif fields.
