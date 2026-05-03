# G4 Checkpoint Approval Resume Plan

## Status
G4 ist als sicherer Kontrollvertrag umgesetzt. Es gibt Stop-after und Resume-Contract-Inspection, aber keinen riskanten Resume-Executor.

## CLI Befehle
Trockenlauf:
```bash
python3 scripts/agent_core_cli.py --idea "..." --script "..." --pipeline-dry-run
```

Stop nach Scene Plan:
```bash
python3 scripts/agent_core_cli.py --idea "..." --script "..." --stop-after scene_plan
```

Stop nach Prompt-Artefakten:
```bash
python3 scripts/agent_core_cli.py --idea "..." --script "..." --stop-after model_prompts
```

Approval Gates aktivieren:
```bash
python3 scripts/agent_core_cli.py --idea "..." --script "..." --approval-gates-enabled
```

Checkpoint anzeigen:
```bash
python3 scripts/agent_core_cli.py --inspect-checkpoints /workspace/agent_runs/<job_id>
```

Approval schreiben:
```bash
python3 scripts/agent_core_cli.py --approve-checkpoint /workspace/agent_runs/<job_id> approve_plan --approved-by "human" --approval-note "reviewed"
```

Reject schreiben:
```bash
python3 scripts/agent_core_cli.py --reject-checkpoint /workspace/agent_runs/<job_id> approve_prompts --rejected-by "human" --approval-note "not good enough"
```

## Stop-after Verhalten
Result-Metadata enthaelt:
- `stopped_after`
- `stop_after_requested`
- `render_started=false`
- `model_backends_started=false`
- `current_checkpoint_id`
- `blocked_by_checkpoint_id`
- `produced_artifacts`
- `next_action`

`--stop-after storyboard` stoppt bewusst vor Storyboard-/Image-Backend-Ausfuehrung. Das ist sicherer als versehentlich Z-Image zu starten.

## Resume Contract
Utility:
- `agent_core/resume_contract.py`
- `inspect_resume_contract(run_dir)`

Regeln:
- Wenn `approve_plan` approved ist, duerfen Plan-Artefakte spaeter wiederverwendet werden, ausser `force_replan`.
- Wenn `approve_prompts` approved ist, duerfen Prompt-Artefakte spaeter wiederverwendet werden, ausser `force_prompts`.
- Wenn Storyboard existiert, braucht Wiederverwendung oder Rerun eine explizite Policy.
- Wenn Takes existieren, darf ein Resume nicht blind doppelt rendern.
- Wenn ein Checkpoint rejected ist, darf der Lauf nicht weitergehen.
- Nie alte Prompts mit neuen Takes mischen, ohne Decision-Log-Eintrag.

## Future Work
Ein echter Resume-Executor braucht Idempotenz, Artefakt-Reuse, Force-Replan/Force-Prompts und klare Regeln fuer Storyboard/Takes. Das wurde absichtlich nicht halb implementiert.
