## title
Quality Review Stage Skill

## purpose
Review generated assets for technical validity, creative fit, and platform readiness.

## when_to_use
Use after storyboard generation, video rendering, selection, and final assembly.

## rules
- Separate technical artifact failures from creative weakness.
- Review against scene intent and platform goals.
- Record evidence in JSON artifacts.

## do
- Check boring scene, weak hook, unclear action, visual incoherence, and composition.
- Track warnings separately from blocking issues.
- Preserve reviewer evidence for diagnosis.

## dont
- Do not approve off-topic but pretty outputs.
- Do not make untraceable selection decisions.
- Do not ignore failed validation reports.

## output_contract
- ReviewPlan records provider, checks, platform_fit_checks, artifact_checks, and selection_policy.

## common_failures
- Qwen review seems wrong.
- Good take is rejected.
- Final verdict misses obvious weakness.

## audit_hints
- Inspect takes.json, storyboard_plan.json, and result.json final_quality_verdict.
