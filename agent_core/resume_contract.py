from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REUSABLE_ARTIFACTS = {
    "plan": "plan.json",
    "director_output": "director_output.json",
    "scene_plan": "scene_plan.json",
    "prompt_audit": "prompt_audit.json",
    "model_prompts": "model_prompts.json",
    "storyboard_plan": "storyboard_plan.json",
    "takes": "takes.json",
    "result": "result.json",
    "decision_log": "decision_log.json",
}

RESUME_RULES = [
    "If approve_plan exists and is approved, plan artifacts may be reused unless force_replan is set.",
    "If approve_prompts exists and is approved, model_prompts may be reused unless force_prompts is set.",
    "If storyboard exists, reuse or rerun must follow a future explicit policy.",
    "If takes exist, a future resume executor must not render duplicate takes blindly.",
    "If a checkpoint has a rejection file, the run must not continue without a new explicit decision.",
    "Never mix old prompts with new takes without a decision_log entry.",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def inspect_resume_contract(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    state = _read_json(root / "state.json")
    checkpoints_payload = _read_json(root / "checkpoints.json")
    checkpoints = checkpoints_payload.get("checkpoints") or state.get("checkpoints") or {}
    approvals_dir = root / "approvals"

    approval_gate_status: dict[str, dict[str, Any]] = {}
    rejected_checkpoints: list[str] = []
    for checkpoint_id, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict):
            continue
        approval_path = approvals_dir / f"{checkpoint_id}.json"
        approval_payload = _read_json(approval_path)
        approved = approval_payload.get("approved")
        if approved is False:
            rejected_checkpoints.append(str(checkpoint_id))
        approval_gate_status[str(checkpoint_id)] = {
            "checkpoint_status": checkpoint.get("status"),
            "approval_required": bool(checkpoint.get("approval_required")),
            "approval_file": str(approval_path),
            "approval_file_exists": approval_path.exists(),
            "approved": approved,
            "approved_by": approval_payload.get("approved_by"),
        }

    reusable_artifacts = {
        key: {
            "path": str(root / filename),
            "exists": (root / filename).exists(),
        }
        for key, filename in REUSABLE_ARTIFACTS.items()
    }

    blocked = state.get("blocked_by_checkpoint_id") or checkpoints_payload.get("blocked_by_checkpoint_id")
    return {
        "run_dir": str(root),
        "resume_supported": False,
        "executor_resume_status": "future_work",
        "current_checkpoint_id": state.get("current_checkpoint_id") or checkpoints_payload.get("current_checkpoint_id"),
        "blocked_by_checkpoint_id": blocked,
        "approval_gate_status": approval_gate_status,
        "rejected_checkpoints": rejected_checkpoints,
        "has_rejection": bool(rejected_checkpoints),
        "can_continue_by_contract": bool(blocked) is False and not rejected_checkpoints,
        "reusable_artifacts": reusable_artifacts,
        "rules": list(RESUME_RULES),
        "next_action": "resume executor is future work; inspect approvals and rerun only under an explicit policy",
    }
