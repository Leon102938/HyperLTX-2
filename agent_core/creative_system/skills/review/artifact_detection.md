## title
Artifact Detection Skill

## purpose
Detect technical and visual artifacts that make generated assets unusable.

## when_to_use
Use in image validation, video validation, Qwen review, and final quality gates.

## rules
- Check generated text, warped objects, bad anatomy, flicker, composition breakage, and wrong aspect.
- Treat readable fake text and UI as high-risk.
- Record specific artifact evidence.

## do
- Validate files and decode metadata before creative judgment.
- Flag phone/UI/screen/collage artifacts.
- Attach related artifact paths.

## dont
- Do not hide technical failures as creative warnings.
- Do not ignore decode or duration mismatch.
- Do not pass assets with visible fake text.

## output_contract
- Review issue categories include artifact, text_artifact, device_artifact, decode_error, and duration_mismatch.

## common_failures
- Visible text.
- Phone/UI in image.
- Split-screen or collage output.

## audit_hints
- Inspect validation fields in takes.json and storyboard_plan.json.
