## title
Postability Review Skill

## purpose
Judge whether the output is good enough to post for the target platform.

## when_to_use
Use at final quality gate and after take selection.

## rules
- A postable clip must have a clear hook, readable action, coherent style, and no obvious artifacts.
- Platform fit is a first-class quality criterion.
- Weak but technically valid clips can need review.

## do
- Check hook, pacing, composition, and visual clarity.
- Flag generic stock feel.
- Record next module to improve when not postable.

## dont
- Do not approve boring outputs.
- Do not ignore target orientation and viewing context.
- Do not rely on voice alone to save visuals.

## output_contract
- Final verdict includes postable boolean or equivalent status with issues and warnings.

## common_failures
- Prompts look clean but output is bad.
- Voice fits but visuals do not.
- Clip has no first-second hook.

## audit_hints
- Inspect result.json final_quality_verdict and model_prompts.json.
