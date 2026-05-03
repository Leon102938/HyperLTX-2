## title
Creative Strategy Stage Skill

## purpose
Convert user intent into a mode, platform, goal, constraints, and flexible motif set.

## when_to_use
Use before beat planning and visual direction.

## rules
- Preserve user intent.
- Select motif families, not fixed mandatory scenes.
- Keep quality risks explicit.

## do
- Identify hook, audience feel, and payoff.
- Record selected mode and style.
- Carry hard bans like visible text and UI.

## dont
- Do not hard-code a single shot sequence when motifs can vary.
- Do not solve model failures with prompt spam.
- Do not ignore platform context.

## output_contract
- CreativeStrategy contains mode_id, style_id, creative_goal, motif_families, constraints, and skill_ids.

## common_failures
- Director goes off-topic.
- Motifs are too rigid.
- Strategy lacks a concrete hook.

## audit_hints
- Check director_output.json and plan.metadata selected_mode/style/skills.
