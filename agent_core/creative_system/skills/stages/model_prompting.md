## title
Model Prompting Stage Skill

## purpose
Compile debug prompts, positive prompts, negative prompts, and backend-specific sent prompts without leakage.

## when_to_use
Use whenever prompt_builder prepares prompts for Z-Image, LTX, or review.

## rules
- Keep debug_prompt human-readable but never send it to image/video backends.
- Keep positive and negative model prompt roles separate.
- Respect backend prompt policy.

## do
- Trace zimage_prompt_sent and ltx prompt fields.
- Warn if fallback/debug prompt would reach a backend.
- Keep Z-Image positive-only.

## dont
- Do not merge long negative lists into positive image prompts.
- Do not include script sentences in backend prompts.
- Do not hide policy decisions.

## output_contract
- ModelPromptPlan includes backend_prompt_policy, positive_model_prompt, negative_model_prompt, sent prompts, warnings, and skill_ids.

## common_failures
- Debug labels leak to backend.
- Negative prompt spam harms image generation.
- LTX prompt is too long to follow.

## audit_hints
- Inspect prompt_audit.json and model_prompts.json checks.
