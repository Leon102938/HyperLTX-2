# G5 Quality Decision Log

## Status
G5 staerkt Creative Quality Review und Decision Log, ohne echte Modelle zu laden.

## Creative Quality Review
Neue metadata-only Utility:
- `evaluate_creative_quality_metadata()`

Sie prueft Hinweise auf:
- boring scene
- weak hook
- unclear action
- generic stock feel
- physical incoherence
- bad composition
- poor platform fit
- no visual change
- dead/static scene
- confusing subject
- mismatch between voice/script and visuals

Wichtig: Diese Heuristik setzt `real_vlm_inference_used=false`. Sie behauptet keine echte Sichtpruefung.

## Final Quality Verdict
`evaluate_final_quality_verdict()` kann jetzt aufnehmen:
- `creative_quality_warnings`
- `platform_fit_warnings`

Damit kann ein technisch sauberer, aber langweiliger oder plattformschwacher Clip spaeter `needs_review` werden.

## Qwen3-VL Reviewer
Der Reviewer-Systemprompt fordert weiterhin JSON-only und wurde um kreative Qualitaetskriterien erweitert. Real inference wird durch G5 nicht gestartet.

## Decision Log
`decision_log.json` protokolliert jetzt besser:
- selected pipeline
- selected mode
- selected style
- loaded/missing skills
- selected motif families
- selected shot recipes
- backend prompt policy
- approval gate status
- stop_after
- quality_decision Contract

## Future Work
- Echte `selected_take`-Entscheidung nach Selection in `decision_log.json` append-only loggen.
- Echte `final_quality_decision` nach Assembly/Final Gate append-only loggen.
- Reviewer-Ausgaben mit Stage Contracts verbinden.
- Provider/Tool Selector fuer Qwen/heuristic/human review ergaenzen.
