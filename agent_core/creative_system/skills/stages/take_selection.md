## title
Take Selection Stage Skill

## purpose
Select the best valid take using validation, review, creative fit, and fallback policy.

## when_to_use
Use after rendering multiple takes or storyboard candidates.

## rules
- Prefer valid takes that match visual goal and platform hook.
- Keep fallback selection explicit.
- Log why a take won.

## do
- Compare technical status, review status, and scene intent.
- Record selected_take and selection reason.
- Preserve rejected alternatives for diagnosis.

## dont
- Do not select first success blindly when better valid takes exist.
- Do not hide fallback decisions.
- Do not confuse technical pass with creative pass.

## output_contract
- Selection result records selected_take, decision reason, fallback status, and related artifacts.

## common_failures
- Good take not selected.
- Selected take is technically valid but boring.
- Retry take overrides a better original.

## audit_hints
- Inspect takes.json selected flags, review_status, and metadata.
