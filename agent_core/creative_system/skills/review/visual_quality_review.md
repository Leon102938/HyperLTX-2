## title
Visual Quality Review Skill

## purpose
Review generated visuals for clarity, composition, artifacts, and match to intent.

## when_to_use
Use for keyframe, take, and final review stages.

## rules
- Review against the planned scene and platform, not generic beauty alone.
- Record evidence and severity.
- Separate warnings from blocking failures.

## do
- Check subject clarity, composition, visible action, and coherence.
- Flag artifacts, text, UI, phone, collage, and off-topic motifs.
- Preserve artifact paths in related_artifacts.

## dont
- Do not pass a scene just because it is technically valid.
- Do not reject creative variation without evidence.
- Do not hide review uncertainty.

## output_contract
- Review returns status, score, issues, warnings, and evidence.

## common_failures
- Qwen review seems wrong.
- Final quality verdict misses the visible problem.
- Bad composition passes.

## audit_hints
- Compare review output with final.mp4/frames and scene_plan.json.
