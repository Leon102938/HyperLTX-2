# G8 Feedback Policy Scaffold

## Zweck
Das G8 Scaffold uebersetzt Review-Probleme in strukturierte Handlungsvorschlaege. Es fuehrt noch keinen Retry aus und startet keine Render- oder Modellstufen.

## Modul
`agent_core/feedback_policy.py` definiert:
- `FeedbackAction`
- `suggest_feedback_actions(review)`

`FeedbackAction` enthaelt:
- `action_type`
- `target_stage`
- `reason`
- `suggested_fix`
- `blocking`
- `retry_budget_impact`

## Mapping
- `visible_text`, `ui`, `phone`, `screen`: Keyframe/Prompt ablehnen oder strenger regenerieren.
- `boring_scene`, `dead_static`: Szene neu planen oder alternativen Beat Candidate waehlen.
- `weak_hook`: staerkeren Hook Candidate waehlen.
- `unclear_action`: Aktion vereinfachen oder enger framen.
- `generic_stock_feel`: spezifischeres taktiles physisches Detail nutzen.
- `physical_incoherence`: Objekt-/Human-Action vereinfachen.
- `low_phone_size_readability`: groesseres Subject, weniger Objekte, klarere Komposition.
- `voice_visual_mismatch`: Beat oder Narration-Mapping anpassen.

## DecisionLog
`decision_log.py` kann FeedbackAction-Vorschlaege persistieren, wenn sie in Plan-Metadaten unter `feedback_actions` vorhanden sind. Das ist bewusst nur Trace und Policy, kein Executor.

## Future Work
Der echte G8-Schritt ist ein kontrollierter Feedback Loop / Retry Executor mit Retry-Budget, Blocking-Regeln, Stop-Bedingungen und klaren Artefaktvertraegen.
