## title
Negative Prompt Policy Skill

## purpose
Keep negative prompting short, separated, and backend-aware.

## when_to_use
Use whenever compiling model prompts or auditing backend prompt policy.

## rules
- Use negative prompts only where the backend supports or accepts them.
- Keep negative lists compact and high-risk.
- Never send negative spam to HiDream-O1-Dev.

## do
- Include text/UI/device/collage bans when relevant.
- Record whether the backend has a separate negative_prompt field.
- Warn when fallback combines negative terms into a positive channel.

## dont
- Do not duplicate negative terms.
- Do not put positive constraints in negative prompt.
- Do not expand avoid lists as a substitute for better visual direction.

## output_contract
- backend_prompt_policy documents hidream and ltx behavior.
- ltx_negative_prompt_sent is present when planned or supported.

## common_failures
- Negative prompt spam reduces prompt clarity.
- Positive constraints are negated.
- Image model receives avoid list.

## audit_hints
- Check no_repeated_forbidden_spam and hidream_positive_only_applied.
