## title
Motion And Camera Skill

## purpose
Make every planned scene contain controlled motion and clear camera intent.

## when_to_use
Use for beat planning, visual direction, LTX prompting, and take review.

## rules
- Every video beat needs visible motion or state change.
- Camera motion must support the action.
- Avoid motion that creates blur or incoherence.

## do
- Use gentle push-in, locked close-up, hand action, reveal, or reset gesture.
- Keep motion simple and observable.
- Match scene duration to action complexity.

## dont
- Do not ask for complex choreography in short clips.
- Do not use vague cinematic movement without subject action.
- Do not hide the subject.

## output_contract
- Scene and take prompts include motion_language and camera intent.

## common_failures
- Good image but bad video.
- Static scene.
- Unclear action after generation.

## audit_hints
- Check ltx_positive_prompt_sent and take review reasons.
