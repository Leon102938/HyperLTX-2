## title
Boring Scene Detection Skill

## purpose
Detect scenes that are clean but too static, generic, or low-signal.

## when_to_use
Use during planning review, take review, and final quality verdict.

## rules
- Boring means weak action, weak change, weak hook, or generic stock feel.
- Calm can pass only if it contains a visible intentional action.
- The first scene has stricter hook expectations.

## do
- Look for visible action or before/after change.
- Compare scene role to what the clip shows.
- Flag repeated static setups.

## dont
- Do not confuse minimal with boring if action is clear.
- Do not approve empty ambience as payoff.
- Do not add text as the default fix.

## output_contract
- Review issues can include boring_scene, weak_hook, no_visible_change, and generic_stock_feel.

## common_failures
- Clip is boring.
- Final quality verdict is too generous.
- Take selection prefers a pretty static take.

## audit_hints
- Inspect takes.json review issues and scene_plan.json visual_goal/action.
