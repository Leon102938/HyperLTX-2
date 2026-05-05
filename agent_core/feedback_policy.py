from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTION_TYPES = {
    "choose_alternate_beat_candidate",
    "replan_scene",
    "regenerate_keyframe",
    "rerender_take",
    "choose_alternate_take",
    "tighten_prompt",
    "simplify_scene",
    "human_review",
    "stop",
}

TECHNICAL_ISSUES = {"technical_failure", "decode_failed", "missing_artifact", "validation_failed"}
BLOCKING_ISSUES = {"visible_text", "fake_text", "typography", "phone", "ui", "screen", "app", "website"}


@dataclass(frozen=True)
class FeedbackAction:
    action_id: str
    issue_type: str
    action_type: str
    target_stage: str
    target_scene_id: str | None
    reason: str
    suggested_fix: str
    blocking: bool
    retry_budget_impact: str
    confidence: float
    source_review_provider: str
    source_review_real_vlm: bool
    related_checkpoint_id: str
    target_take_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"unknown feedback action_type: {self.action_type}")
        if not self.action_id:
            raise ValueError("FeedbackAction.action_id is required")
        if not self.issue_type:
            raise ValueError("FeedbackAction.issue_type is required")
        if not self.target_stage:
            raise ValueError("FeedbackAction.target_stage is required")
        if not self.reason:
            raise ValueError("FeedbackAction.reason is required")
        if not self.suggested_fix:
            raise ValueError("FeedbackAction.suggested_fix is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("FeedbackAction.confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "issue_type": self.issue_type,
            "action_type": self.action_type,
            "target_stage": self.target_stage,
            "target_scene_id": self.target_scene_id,
            "target_take_id": self.target_take_id,
            "reason": self.reason,
            "suggested_fix": self.suggested_fix,
            "blocking": self.blocking,
            "retry_budget_impact": self.retry_budget_impact,
            "confidence": round(float(self.confidence), 3),
            "source_review_provider": self.source_review_provider,
            "source_review_real_vlm": bool(self.source_review_real_vlm),
            "related_checkpoint_id": self.related_checkpoint_id,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedbackAction":
        return cls(
            action_id=str(payload.get("action_id") or ""),
            issue_type=str(payload.get("issue_type") or ""),
            action_type=str(payload.get("action_type") or ""),
            target_stage=str(payload.get("target_stage") or ""),
            target_scene_id=_optional_str(payload.get("target_scene_id")),
            target_take_id=_optional_str(payload.get("target_take_id")),
            reason=str(payload.get("reason") or ""),
            suggested_fix=str(payload.get("suggested_fix") or ""),
            blocking=bool(payload.get("blocking")),
            retry_budget_impact=str(payload.get("retry_budget_impact") or "none"),
            confidence=float(payload.get("confidence") if payload.get("confidence") is not None else 0.0),
            source_review_provider=str(payload.get("source_review_provider") or "unknown"),
            source_review_real_vlm=bool(payload.get("source_review_real_vlm")),
            related_checkpoint_id=str(payload.get("related_checkpoint_id") or "feedback_review"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RetryBudget:
    max_keyframe_retries_per_scene: int = 1
    max_video_retries_per_scene: int = 1
    max_plan_retries: int = 1
    used_retries: dict[str, int] = field(default_factory=dict)

    def remaining_retries(self) -> dict[str, int]:
        return {
            "keyframe": max(0, self.max_keyframe_retries_per_scene - int(self.used_retries.get("keyframe", 0))),
            "video": max(0, self.max_video_retries_per_scene - int(self.used_retries.get("video", 0))),
            "plan": max(0, self.max_plan_retries - int(self.used_retries.get("plan", 0))),
        }

    def exhausted(self) -> bool:
        remaining = self.remaining_retries()
        return all(value <= 0 for value in remaining.values())

    def can_spend(self, impact: str) -> bool:
        remaining = self.remaining_retries()
        if impact == "spend_keyframe_retry":
            return remaining["keyframe"] > 0
        if impact == "spend_take_retry":
            return remaining["video"] > 0
        if impact == "spend_replan_budget":
            return remaining["plan"] > 0
        if impact in {"none", "choose_existing_take"}:
            return True
        if impact == "spend_keyframe_or_take_retry":
            return remaining["keyframe"] > 0 or remaining["video"] > 0
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_keyframe_retries_per_scene": self.max_keyframe_retries_per_scene,
            "max_video_retries_per_scene": self.max_video_retries_per_scene,
            "max_plan_retries": self.max_plan_retries,
            "used_retries": dict(self.used_retries),
            "remaining_retries": self.remaining_retries(),
            "exhausted": self.exhausted(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetryBudget":
        payload = dict(payload or {})
        return cls(
            max_keyframe_retries_per_scene=int(payload.get("max_keyframe_retries_per_scene", 1)),
            max_video_retries_per_scene=int(payload.get("max_video_retries_per_scene", 1)),
            max_plan_retries=int(payload.get("max_plan_retries", 1)),
            used_retries=dict(payload.get("used_retries") or {}),
        )


@dataclass(frozen=True)
class RetryPlan:
    feedback_actions: list[FeedbackAction]
    allowed_next_actions: list[str]
    blocked: bool
    requires_human_approval: bool
    reusable_artifacts: list[str]
    invalidated_artifacts: list[str]
    reason: str
    retry_budget: RetryBudget
    top_priority_action: FeedbackAction | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_actions": [action.to_dict() for action in self.feedback_actions],
            "top_priority_action": self.top_priority_action.to_dict() if self.top_priority_action else None,
            "allowed_next_actions": list(self.allowed_next_actions),
            "blocked": self.blocked,
            "requires_human_approval": self.requires_human_approval,
            "reusable_artifacts": list(self.reusable_artifacts),
            "invalidated_artifacts": list(self.invalidated_artifacts),
            "reason": self.reason,
            "retry_budget": self.retry_budget.to_dict(),
        }


def suggest_feedback_actions(review: dict[str, Any] | list[str]) -> list[FeedbackAction]:
    evaluation = evaluate_feedback_actions(review, {}, {})
    return evaluation["feedback_actions"]


def evaluate_feedback_actions(
    review_payload: dict[str, Any] | list[str],
    stage_contracts: dict[str, Any] | None = None,
    decision_log_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = _extract_issues(review_payload)
    provider, real_vlm = _review_source(review_payload)
    actions = [
        _action_for_issue(issue, index=index, provider=provider, real_vlm=real_vlm)
        for index, issue in enumerate(issues, start=1)
    ]
    actions = [action for action in actions if action is not None]
    actions = _dedupe(actions)
    actions.sort(key=_priority_key)
    top = actions[0] if actions else None
    budget = RetryBudget.from_dict((decision_log_context or {}).get("retry_budget"))
    plan = build_retry_plan(actions, retry_budget=budget, stage_contracts=stage_contracts or {})
    should_block = any(action.blocking for action in actions) or plan.blocked
    return {
        "feedback_actions": actions,
        "feedback_actions_json": [action.to_dict() for action in actions],
        "top_priority_action": top,
        "top_priority_action_json": top.to_dict() if top else None,
        "recommended_next_stage": top.target_stage if top else "pass",
        "should_block": should_block,
        "retry_budget_summary": budget.to_dict(),
        "retry_plan": plan,
        "retry_plan_json": plan.to_dict(),
        "source_review_provider": provider,
        "source_review_real_vlm": real_vlm,
    }


def build_retry_plan(
    feedback_actions: list[FeedbackAction],
    *,
    retry_budget: RetryBudget | None = None,
    stage_contracts: dict[str, Any] | None = None,
) -> RetryPlan:
    retry_budget = retry_budget or RetryBudget()
    allowed: list[str] = []
    invalidated: set[str] = set()
    reusable: set[str] = {"input_job", "skill_injection_context", "decision_log"}
    requires_human = False
    blocked = False
    reasons: list[str] = []
    sorted_actions = sorted(feedback_actions, key=_priority_key)
    for action in sorted_actions:
        if action.action_type in {"human_review", "stop"}:
            requires_human = True
            blocked = True
        if action.blocking:
            blocked = True
        if not retry_budget.can_spend(action.retry_budget_impact):
            requires_human = True
            blocked = True
            allowed.append("human_review")
            reasons.append(f"retry budget exhausted for {action.issue_type}")
            continue
        allowed.append(action.action_type)
        invalidated.update(_invalidated_artifacts_for_action(action))
        reusable.update(_reusable_artifacts_for_action(action))
        reasons.append(f"{action.issue_type}: {action.suggested_fix}")
    if retry_budget.exhausted() and sorted_actions:
        requires_human = True
        blocked = True
        if "human_review" not in allowed:
            allowed.append("human_review")
    allowed = _unique_strings(allowed)
    top = sorted_actions[0] if sorted_actions else None
    return RetryPlan(
        feedback_actions=sorted_actions,
        top_priority_action=top,
        allowed_next_actions=allowed,
        blocked=blocked,
        requires_human_approval=requires_human or blocked,
        reusable_artifacts=sorted(reusable),
        invalidated_artifacts=sorted(invalidated),
        reason="; ".join(reasons) if reasons else "no feedback action required",
        retry_budget=retry_budget,
    )


def build_feedback_checkpoint_state(evaluation: dict[str, Any]) -> dict[str, Any]:
    top = evaluation.get("top_priority_action")
    top_payload = top.to_dict() if isinstance(top, FeedbackAction) else evaluation.get("top_priority_action_json")
    return {
        "blocked_by_feedback_action_id": (top_payload or {}).get("action_id"),
        "feedback_next_action": (top_payload or {}).get("action_type"),
        "feedback_requires_approval": bool(evaluation.get("should_block") or (evaluation.get("retry_plan_json") or {}).get("requires_human_approval")),
        "recommended_next_stage": evaluation.get("recommended_next_stage"),
        "suggested_fix": (top_payload or {}).get("suggested_fix"),
    }


def _action_for_issue(issue: dict[str, Any], *, index: int, provider: str, real_vlm: bool) -> FeedbackAction | None:
    issue_type = _normalize_issue_type(str(issue.get("issue_type") or issue.get("issue") or issue.get("type") or "unknown"))
    scene_id = _optional_str(issue.get("scene_id") or issue.get("target_scene_id"))
    take_id = _optional_str(issue.get("take_id") or issue.get("target_take_id"))
    reason = str(issue.get("reason") or issue.get("message") or issue_type)
    action_type, target_stage, suggested_fix, blocking, impact, confidence = _mapping(issue_type)
    return FeedbackAction(
        action_id=f"feedback_{index:03d}_{issue_type}",
        issue_type=issue_type,
        action_type=action_type,
        target_stage=target_stage,
        target_scene_id=scene_id,
        target_take_id=take_id,
        reason=reason,
        suggested_fix=suggested_fix,
        blocking=blocking,
        retry_budget_impact=impact,
        confidence=float(issue.get("confidence") if issue.get("confidence") is not None else confidence),
        source_review_provider=str(issue.get("provider") or provider),
        source_review_real_vlm=bool(issue.get("real_vlm_inference_used") if issue.get("real_vlm_inference_used") is not None else real_vlm),
        related_checkpoint_id=str(issue.get("related_checkpoint_id") or "feedback_review"),
        metadata={k: v for k, v in issue.items() if k not in {"issue_type", "issue", "type", "scene_id", "take_id", "reason", "message"}},
    )


def _mapping(issue_type: str) -> tuple[str, str, str, bool, str, float]:
    if issue_type in {"visible_text", "fake_text", "typography"}:
        return ("regenerate_keyframe", "storyboard", "reject text-bearing visual and regenerate with clean unlabeled physical scene", True, "spend_keyframe_retry", 0.9)
    if issue_type in {"phone", "ui", "screen", "app", "website"}:
        return ("regenerate_keyframe", "storyboard", "remove device/UI framing and replan as a physical non-screen scene", True, "spend_keyframe_retry", 0.88)
    if issue_type in {"boring_scene", "dead_static_scene", "no_visual_change"}:
        return ("choose_alternate_beat_candidate", "beat_plan", "use stronger visible action, tactile detail, or motion-first candidate", False, "spend_replan_budget", 0.78)
    if issue_type == "weak_hook":
        return ("choose_alternate_beat_candidate", "beat_plan", "prefer tactile_first or motion_first hook candidate", False, "spend_replan_budget", 0.82)
    if issue_type == "unclear_action":
        return ("simplify_scene", "visual_direction", "use closer framing and one clear physical action", False, "spend_take_retry", 0.78)
    if issue_type == "generic_stock_feel":
        return ("replan_scene", "beat_plan", "add specific tactile physical detail and remove generic lifestyle staging", False, "spend_replan_budget", 0.76)
    if issue_type == "physical_incoherence":
        return ("simplify_scene", "visual_direction", "use fewer objects and an easier human action", False, "spend_take_retry", 0.82)
    if issue_type == "low_phone_size_readability":
        return ("tighten_prompt", "model_prompting", "larger subject, fewer objects, close or medium framing", False, "spend_keyframe_or_take_retry", 0.8)
    if issue_type == "voice_visual_mismatch":
        return ("replan_scene", "beat_plan", "align beat role with narration intent before rendering again", False, "spend_replan_budget", 0.78)
    if issue_type == "bad_composition":
        return ("regenerate_keyframe", "storyboard", "use clearer subject anchor and simpler composition", False, "spend_keyframe_retry", 0.76)
    if issue_type in TECHNICAL_ISSUES:
        return ("stop", "technical_validation", "stop automatic retry and inspect failed technical artifact", True, "none", 0.92)
    return ("human_review", "quality_review", "unknown issue type; inspect manually before retry", True, "none", 0.45)


def _extract_issues(review_payload: dict[str, Any] | list[str]) -> list[dict[str, Any]]:
    if isinstance(review_payload, list):
        return [_issue_from_value(item) for item in review_payload]
    issues: list[dict[str, Any]] = []
    for key in ("issues", "warnings", "creative_quality_warnings", "platform_fit_warnings", "artifact_warnings"):
        for item in review_payload.get(key) or []:
            issue = _issue_from_value(item)
            if key == "creative_quality_warnings" and "source" not in issue:
                issue["source"] = key
            issues.append(issue)
    for scene in review_payload.get("scene_reviews") or []:
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("scene_id")
        for key in ("issues", "warnings", "creative_quality_warnings", "platform_fit_warnings"):
            for item in scene.get(key) or []:
                issue = _issue_from_value(item)
                issue.setdefault("scene_id", scene_id)
                issues.append(issue)
    return issues


def _issue_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload["issue_type"] = _normalize_issue_type(str(payload.get("issue_type") or payload.get("issue") or payload.get("type") or payload.get("reason") or "unknown"))
        return payload
    text = str(value)
    return {"issue_type": _normalize_issue_type(text), "reason": text}


def _normalize_issue_type(value: str) -> str:
    lower = value.lower().strip().replace("-", "_").replace(" ", "_")
    parts = {part for part in lower.split("_") if part}
    aliases = {
        "readable_text": "visible_text",
        "text": "visible_text",
        "visible_type": "visible_text",
        "dead_static": "dead_static_scene",
        "static_scene": "dead_static_scene",
        "generic_stock": "generic_stock_feel",
        "poor_platform_fit": "low_phone_size_readability",
        "device": "phone",
        "website_ui": "website",
    }
    for token, issue_type in aliases.items():
        if ("_" in token and token in lower) or (token in parts):
            return issue_type
    for known in [
        "low_phone_size_readability",
        "visible_text",
        "fake_text",
        "typography",
        "phone",
        "ui",
        "screen",
        "app",
        "website",
        "boring_scene",
        "dead_static_scene",
        "no_visual_change",
        "weak_hook",
        "unclear_action",
        "generic_stock_feel",
        "physical_incoherence",
        "voice_visual_mismatch",
        "bad_composition",
        *sorted(TECHNICAL_ISSUES),
    ]:
        if ("_" in known and known in lower) or (known in parts):
            return known
    return lower or "unknown"


def _review_source(review_payload: dict[str, Any] | list[str]) -> tuple[str, bool]:
    if not isinstance(review_payload, dict):
        return "manual_issue_list", False
    provider = str(review_payload.get("provider") or review_payload.get("review_provider") or review_payload.get("source_review_provider") or "heuristic")
    real_vlm = bool(review_payload.get("real_vlm_inference_used") or review_payload.get("source_review_real_vlm"))
    return provider, real_vlm


def _priority_key(action: FeedbackAction) -> tuple[int, str]:
    if action.issue_type in TECHNICAL_ISSUES or action.action_type == "stop":
        return (0, action.action_id)
    if action.issue_type in BLOCKING_ISSUES:
        return (1, action.action_id)
    if action.blocking:
        return (2, action.action_id)
    if action.issue_type in {"weak_hook", "boring_scene", "dead_static_scene", "no_visual_change"}:
        return (3, action.action_id)
    return (4, action.action_id)


def _invalidated_artifacts_for_action(action: FeedbackAction) -> list[str]:
    scene = action.target_scene_id or "*"
    if action.action_type in {"tighten_prompt", "regenerate_keyframe"}:
        return [f"storyboard/{scene}", f"keyframes/{scene}", f"takes/{scene}", f"model_prompts/{scene}"]
    if action.action_type in {"replan_scene", "simplify_scene", "choose_alternate_beat_candidate"}:
        return [f"scene_plan/{scene}", f"model_prompts/{scene}", f"storyboard/{scene}", f"takes/{scene}"]
    if action.action_type == "rerender_take":
        return [f"takes/{scene}"]
    if action.action_type == "choose_alternate_take":
        return []
    return []


def _reusable_artifacts_for_action(action: FeedbackAction) -> list[str]:
    if action.action_type == "choose_alternate_take":
        return ["scene_plan", "model_prompts", "storyboard", "existing_valid_takes"]
    return ["input_job", "skill_injection_context", "creative_intent"]


def _dedupe(actions: list[FeedbackAction]) -> list[FeedbackAction]:
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[FeedbackAction] = []
    for action in actions:
        key = (action.issue_type, action.target_scene_id, action.target_take_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
