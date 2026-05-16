## title
Positive Image Prompting Skill

## purpose
Create positive-only image prompts that describe exactly what should be visible.

## when_to_use
Use before sending prompts to image/storyboard backends.

## rules
- Describe desired visual facts only.
- Keep forbidden concepts out of positive text.
- Keep prompt short.

## do
- Include subject, action, setting, light, framing, style.
- Use "single full-frame" when layout risk is high.
- Use countable props.

## dont
- Do not include "no", "avoid", or forbidden objects as positive prompt content.
- Do not include debug labels or script sentences.
- Do not ask for typography.

## output_contract
- positive_model_prompt is usable as hidream_prompt_sent.
- hidream_prompt_sent has no Avoid section.

## common_failures
- Prompt mentions forbidden object and model draws it.
- Prompt is too abstract.
- Prompt includes visible text instructions.

## audit_hints
- Check positive_risky_terms_detected in prompt_audit.json.
