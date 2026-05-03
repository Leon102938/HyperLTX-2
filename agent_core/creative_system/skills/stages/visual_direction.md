## title
Visual Direction Stage Skill

## purpose
Define the physical world, camera language, lighting, and visual bans before model prompting.

## when_to_use
Use after beat planning and before prompt compilation.

## rules
- Be physical and visual.
- Preserve style lock across scenes.
- Keep forbidden visuals out of positive content.

## do
- Describe allowed props, environment, action, camera, and lighting.
- Keep one coherent visual identity.
- Mark text/UI/device risks.

## dont
- Do not include script imperatives in model prompts.
- Do not add logos, labels, phones, UI, or screens as positive props.
- Do not create collage layouts for normal scenes.

## output_contract
- VisualDirection includes identity, lighting, camera_language, allowed_visuals, forbidden_visuals, and skill_ids.

## common_failures
- Wrong motif appears.
- Visible text or UI appears.
- Style varies scene to scene.

## audit_hints
- Compare director_output.json style_lock with prompt_audit.json scene contracts.
