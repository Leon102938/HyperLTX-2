## title
Video Prompting Skill

## purpose
Create concise video prompts with action, continuity, camera motion, and lighting.

## when_to_use
Use for LTX text-to-video and image-conditioned video prompts.

## rules
- State the visible action first.
- Keep one scene continuous unless requested otherwise.
- Use short camera and motion cues.

## do
- Include subject, action, environment, camera, light, and duration feel.
- Keep negative terms short and separate where possible.
- Preserve scene intent.

## dont
- Do not pack multiple unrelated actions into one short take.
- Do not include debug labels.
- Do not rely on hidden narration.

## output_contract
- ltx_positive_prompt_sent is concise and physical.
- ltx_negative_prompt_sent is short or explicitly marked unsupported.

## common_failures
- Video drifts away from keyframe.
- Motion is weak.
- Prompt is too long.

## audit_hints
- Check model_prompts.json ltx fields and word counts.
