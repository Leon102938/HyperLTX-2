# G8 Feedback Loop / Retry Policy

## Zweck
G8 macht Review-Ergebnisse handlungsfaehig, ohne echte Retry-Jobs zu starten. Der Vertrag lautet:

Problem erkannt -> FeedbackAction -> Zielstage -> Suggested Fix -> RetryBudget -> RetryPlan -> DecisionLog -> Checkpoint/Human Review.

## Issue zu FeedbackAction
`agent_core/feedback_policy.py` mappt Issues defensiv:
- `visible_text`, `fake_text`, `typography` -> `regenerate_keyframe`, blocking.
- `phone`, `ui`, `screen`, `app`, `website` -> `regenerate_keyframe`, blocking.
- `boring_scene`, `dead_static_scene`, `no_visual_change` -> `choose_alternate_beat_candidate`.
- `weak_hook` -> `choose_alternate_beat_candidate`, bevorzugt tactile/motion-first.
- `unclear_action` -> `simplify_scene`.
- `generic_stock_feel` -> `replan_scene`.
- `physical_incoherence` -> `simplify_scene`.
- `low_phone_size_readability` -> `tighten_prompt`.
- `voice_visual_mismatch` -> `replan_scene`.
- `bad_composition` -> `regenerate_keyframe`.
- unbekannt -> `human_review`.

Jede FeedbackAction enthaelt Action-ID, Issue-Type, Action-Type, Zielstage, Szene/Take, Reason, Suggested Fix, Blocking, Retry-Budget-Impact, Confidence, Review-Provider, Real-VLM-Flag und Checkpoint-ID.

## Evaluator
`evaluate_feedback_actions(review_payload, stage_contracts, decision_log_context)` akzeptiert:
- Take Visual Review
- Final Quality Verdict
- Heuristic Metadata
- Qwen3-VL Review Payload
- manuelle Issue-Listen

Prioritaet:
- technische Fehler vor kreativen Warnungen
- visible text/UI/device vor boring/static
- pro Szene werden eigene Actions erzeugt
- Heuristik bleibt Heuristik; `source_review_real_vlm` wird nicht gefaelscht

## RetryBudget
`RetryBudget` definiert:
- `max_keyframe_retries_per_scene`
- `max_video_retries_per_scene`
- `max_plan_retries`
- `used_retries`
- `remaining_retries`
- `exhausted`

Wenn ein benoetigtes Budget erschoepft ist, verlangt der RetryPlan Human Review oder Stop.

## RetryPlan
`RetryPlan` enthaelt:
- `feedback_actions`
- `top_priority_action`
- `allowed_next_actions`
- `blocked`
- `requires_human_approval`
- `reusable_artifacts`
- `invalidated_artifacts`
- `reason`
- `retry_budget`

Idempotenzregeln:
- Prompt-Aenderung invalidiert Keyframes/Takes/Model-Prompts der Szene.
- BeatPlan-Aenderung invalidiert ScenePlan/ModelPrompts/Storyboard/Takes der Szene.
- `choose_alternate_take` invalidiert keine Prompts.
- Human Review blockiert.
- Alte Prompts duerfen nicht mit neuen Takes gemischt werden, ohne DecisionLog.

## DecisionLog und Checkpoint Trace
DecisionLog kann folgende Entscheidungen schreiben:
- `feedback_action_created`
- `retry_plan_created`
- `blocked_by_feedback`
- `human_review_required`
- `artifact_invalidated`

Checkpoint-kompatible Felder:
- `blocked_by_feedback_action_id`
- `feedback_next_action`
- `feedback_requires_approval`
- `recommended_next_stage`
- `suggested_fix`

## CLI Inspect
`scripts/agent_core_cli.py --inspect-run <run>` liest, falls vorhanden:
- `feedback_actions.json`
- `retry_plan.json`

Inspect zeigt Top Action, Issue, Scene, Blocking, Recommended Next Stage, Suggested Fix, RetryPlan-Status, Allowed Actions und invalidierte Artefakte.

## Smoke Fixture
Pfad:
- `/workspace/agent_runs/g8-feedback-policy-smoke`

Artefakte:
- `sample_review_payload.json`
- `feedback_actions.json`
- `retry_plan.json`
- `decision_log.json`
- `state.json`
- `checkpoints.json`

Der Fixture-Run hat kein `final.mp4`.

## Future Work
- Kein echter Retry Executor in G8.
- G9 soll einen kontrollierten ersten echten V1-Run ausfuehren und die FeedbackActions danach manuell auswerten.
- Ein spaeterer G9/G10-Executor muss Approval Gates, Retry-Budget, Artifact-Invalidation und Resume-Vertrag gemeinsam respektieren.
