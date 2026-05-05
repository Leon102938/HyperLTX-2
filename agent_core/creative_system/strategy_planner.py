from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.prompt_builder import PromptBuilder
from agent_core.schemas import DirectorOutput, JobInput


@dataclass(frozen=True)
class CreativeIntent:
    raw_user_idea: str
    raw_script: str
    sanitized_visual_intent: str
    topic: str
    platform: str
    desired_emotion: str
    content_promise: str
    audience_value: str
    visual_energy: str
    pacing_type: str
    risk_profile: list[str]
    constraints: list[str]
    inferred_mode: str
    inferred_style: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_user_idea": self.raw_user_idea,
            "raw_script": self.raw_script,
            "sanitized_visual_intent": self.sanitized_visual_intent,
            "topic": self.topic,
            "platform": self.platform,
            "desired_emotion": self.desired_emotion,
            "content_promise": self.content_promise,
            "audience_value": self.audience_value,
            "visual_energy": self.visual_energy,
            "pacing_type": self.pacing_type,
            "risk_profile": list(self.risk_profile),
            "constraints": list(self.constraints),
            "inferred_mode": self.inferred_mode,
            "inferred_style": self.inferred_style,
        }


@dataclass(frozen=True)
class BeatCandidate:
    scene_index: int
    role: str
    motif_family: str
    motif_id: str
    shot_recipe_id: str
    visible_action: str
    expected_visual_change: str
    camera_language: str
    allowed_visuals: list[str]
    avoid_risks: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "role": self.role,
            "motif_family": self.motif_family,
            "motif_id": self.motif_id,
            "shot_recipe_id": self.shot_recipe_id,
            "visible_action": self.visible_action,
            "expected_visual_change": self.expected_visual_change,
            "camera_language": self.camera_language,
            "allowed_visuals": list(self.allowed_visuals),
            "avoid_risks": list(self.avoid_risks),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class BeatPlanCandidate:
    candidate_id: str
    hook_pattern: str
    beat_sequence: list[BeatCandidate]
    scene_roles: dict[str, str]
    motif_families: list[str]
    shot_recipes: list[str]
    continuity_strategy: str
    platform_fit_intent: str
    expected_visual_change: list[str]
    risk_notes: list[str]
    score_breakdown: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hook_pattern": self.hook_pattern,
            "beat_sequence": [beat.to_dict() for beat in self.beat_sequence],
            "scene_roles": dict(self.scene_roles),
            "motif_families": list(self.motif_families),
            "shot_recipes": list(self.shot_recipes),
            "continuity_strategy": self.continuity_strategy,
            "platform_fit_intent": self.platform_fit_intent,
            "expected_visual_change": list(self.expected_visual_change),
            "risk_notes": list(self.risk_notes),
            "score_breakdown": dict(self.score_breakdown),
            "rationale": self.rationale,
        }


def analyze_creative_intent(
    *,
    job: JobInput,
    mode: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    mode_id: str = "",
    style_id: str = "",
) -> CreativeIntent:
    mode = dict(mode or {})
    style = dict(style or {})
    combined = f"{job.idea} {job.script}".lower()
    topic = "calm morning productivity" if _contains_any(combined, ["morning", "morgen", "reset", "focus", "fokus"]) else "shortform visual idea"
    desired_emotion = str(mode.get("audience_feel") or "clear, focused, visually calm")
    sanitized = _sanitize_visual_intent(job.idea or job.script, job.script)
    content_promise = "quick physical reset" if "reset" in combined or "morning" in combined else "clear visual payoff"
    audience_value = "a simple repeatable action the viewer can understand at phone size"
    risk_profile = _unique_strings(
        [
            "text_ui_device_artifact_risk",
            "generic_stock_feel_risk",
            "weak_hook_risk",
            *list(mode.get("anti_patterns") or []),
        ]
    )
    return CreativeIntent(
        raw_user_idea=job.idea,
        raw_script=job.script,
        sanitized_visual_intent=sanitized,
        topic=topic,
        platform=str(mode.get("platform_default") or job.metadata.get("platform") or "portrait_short"),
        desired_emotion=desired_emotion,
        content_promise=content_promise,
        audience_value=audience_value,
        visual_energy=str((mode.get("pacing") or {}).get("beat_guidance") or "soft but visible motion"),
        pacing_type=str((mode.get("pacing") or {}).get("overall") or "short clear beats"),
        risk_profile=risk_profile,
        constraints=_unique_strings([*list(mode.get("global_forbidden") or []), *list(style.get("principles") or [])]),
        inferred_mode=mode_id or str(mode.get("mode_id") or ""),
        inferred_style=style_id or str(style.get("style_id") or mode.get("visual_style") or ""),
    )


def generate_beat_plan_candidates(
    *,
    creative_intent: CreativeIntent,
    mode: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    skill_context: dict[str, Any] | None = None,
    scene_count: int = 3,
) -> list[BeatPlanCandidate]:
    mode = dict(mode or {})
    style = dict(style or {})
    avoid = _unique_strings(list(mode.get("global_forbidden") or []) + list(skill_context.get("anti_patterns") or []) if isinstance(skill_context, dict) else list(mode.get("global_forbidden") or []))
    recipes = _candidate_templates(avoid)
    candidates: list[BeatPlanCandidate] = []
    for candidate_id, hook, beats, rationale in recipes:
        selected_beats = _fit_scene_count(beats, scene_count)
        scene_roles = {f"scene_{beat.scene_index:02d}": beat.role for beat in selected_beats}
        candidate = BeatPlanCandidate(
            candidate_id=candidate_id,
            hook_pattern=hook,
            beat_sequence=selected_beats,
            scene_roles=scene_roles,
            motif_families=[beat.motif_family for beat in selected_beats],
            shot_recipes=[beat.shot_recipe_id for beat in selected_beats],
            continuity_strategy="one clean physical world with visible micro-change in every scene",
            platform_fit_intent="large readable subject, few objects, one action per phone-sized frame",
            expected_visual_change=[beat.expected_visual_change for beat in selected_beats],
            risk_notes=_unique_strings([*avoid[:8], "avoid dead/static scene", "avoid generic stock b-roll"]),
            rationale=rationale,
        )
        candidates.append(candidate)
    return candidates


def score_beat_plan_candidate(
    candidate: BeatPlanCandidate,
    creative_intent: CreativeIntent,
    skill_context: dict[str, Any] | None = None,
) -> dict[str, float]:
    text = " ".join(
        [
            candidate.hook_pattern,
            " ".join(beat.visible_action for beat in candidate.beat_sequence),
            " ".join(beat.expected_visual_change for beat in candidate.beat_sequence),
            candidate.rationale,
        ]
    ).lower()
    has_action = all(_contains_any(beat.visible_action.lower(), ["opens", "places", "breathes", "moves", "reveals", "stretch", "turns", "pours", "sets"]) for beat in candidate.beat_sequence)
    has_change = all(bool(beat.expected_visual_change.strip()) for beat in candidate.beat_sequence)
    risk_hits = sum(1 for term in ["phone", "screen", "ui", "text", "paper", "label", "logo"] if term in text)
    generic_hits = sum(1 for term in ["generic", "static", "empty room", "stock"] if term in text)
    hook_strength = 0.9 if candidate.beat_sequence and candidate.beat_sequence[0].role in {"hook", "opening_hook"} else 0.55
    visual_clarity = 0.88 if has_change else 0.45
    action_readability = 0.9 if has_action else 0.42
    originality = 0.78 if len(set(candidate.motif_families)) >= min(3, len(candidate.motif_families)) else 0.58
    model_feasibility = max(0.35, 0.9 - 0.08 * risk_hits)
    artifact_risk = max(0.25, 0.88 - 0.12 * risk_hits)
    platform_fit = 0.88 if "phone" in candidate.platform_fit_intent else 0.72
    continuity = 0.84 if candidate.continuity_strategy else 0.5
    anti_boring_score = max(0.25, 0.9 - 0.15 * generic_hits - (0 if has_action else 0.25))
    total = round(
        hook_strength * 0.14
        + visual_clarity * 0.14
        + action_readability * 0.14
        + originality * 0.08
        + model_feasibility * 0.12
        + artifact_risk * 0.12
        + platform_fit * 0.1
        + continuity * 0.08
        + anti_boring_score * 0.08,
        3,
    )
    return {
        "hook_strength": round(hook_strength, 3),
        "visual_clarity": round(visual_clarity, 3),
        "action_readability": round(action_readability, 3),
        "originality": round(originality, 3),
        "model_feasibility": round(model_feasibility, 3),
        "artifact_risk": round(artifact_risk, 3),
        "platform_fit": round(platform_fit, 3),
        "continuity": round(continuity, 3),
        "anti_boring_score": round(anti_boring_score, 3),
        "total_score": total,
    }


def select_best_beat_plan_candidate(
    candidates: list[BeatPlanCandidate],
    creative_intent: CreativeIntent,
    skill_context: dict[str, Any] | None = None,
) -> tuple[BeatPlanCandidate | None, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    best: BeatPlanCandidate | None = None
    best_score = -1.0
    for candidate in candidates:
        score = score_beat_plan_candidate(candidate, creative_intent, skill_context)
        scored.append({"candidate_id": candidate.candidate_id, "score_breakdown": score})
        if score["total_score"] > best_score:
            best = candidate
            best_score = score["total_score"]
    return best, scored


def apply_selected_candidate_to_director_output(
    *,
    director_output: DirectorOutput,
    selected_candidate: BeatPlanCandidate | None,
) -> DirectorOutput:
    if selected_candidate is None:
        return director_output
    updated = director_output.model_copy(deep=True)
    per_scene = {}
    beat_by_scene = {f"scene_{beat.scene_index:02d}": beat for beat in selected_candidate.beat_sequence}
    for intent in updated.scene_intents:
        beat = beat_by_scene.get(intent.scene_id)
        if not beat:
            continue
        intent.narrative_role = "opening_hook" if beat.role == "hook" else ("final_payoff" if beat.role == "payoff" else beat.role)
        intent.hook_focus = beat.visible_action
        intent.visual_goal = f"{beat.expected_visual_change} through {beat.visible_action}"
        intent.shot_intent = beat.visible_action
        intent.prompt_keywords = _unique_strings([*beat.allowed_visuals, beat.motif_family])[:8]
        intent.notes.append(f"G7 selected beat candidate {selected_candidate.candidate_id}.")
        per_scene[intent.scene_id] = build_per_scene_visual_direction(intent.scene_id, beat)
    updated.prompt_guidance.opening_shot = selected_candidate.beat_sequence[0].visible_action if selected_candidate.beat_sequence else updated.prompt_guidance.opening_shot
    updated.metadata["g7_selected_beat_plan_candidate"] = selected_candidate.to_dict()
    updated.metadata["g7_per_scene_visual_direction"] = per_scene
    return updated


def build_per_scene_visual_direction(scene_id: str, beat: BeatCandidate) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "motif_family": beat.motif_family,
        "motif_id": beat.motif_id,
        "shot_recipe": beat.shot_recipe_id,
        "shot_recipe_id": beat.shot_recipe_id,
        "action": beat.visible_action,
        "expected_visual_change": beat.expected_visual_change,
        "camera_language": beat.camera_language,
        "lighting": "soft natural morning light",
        "movement": "one visible physical micro-action",
        "composition_rules": ["single full-frame physical scene", "large readable subject", "few objects"],
        "object_count_policy": "few visible objects, only what the beat needs",
        "human_action_policy": "one clear human or hand action",
        "allowed_visuals": list(beat.allowed_visuals),
        "avoid_risks": list(beat.avoid_risks),
        "rationale": beat.rationale,
    }


def _candidate_templates(avoid: list[str]) -> list[tuple[str, str, list[BeatCandidate], str]]:
    common_avoid = _unique_strings([*avoid, "visible text", "phones", "screens", "UI", "paper", "labels", "logos"])
    return [
        (
            "light_to_action",
            "light_reveal_hook",
            [
                BeatCandidate(1, "hook", "light_reveal", "curtain_light_shift", "curtain_light_reveal", "a hand opens plain fabric curtains and soft morning light enters a clean room", "visible light change across fabric and wall", "slow push-in with clear silhouette", ["plain fabric curtains", "blank wall", "soft window light"], common_avoid, "open with a visible light change instead of text"),
                BeatCandidate(2, "tactile_detail", "tactile_object_detail", "water_glass_empty_table", "water_glass_closeup", "one clear water glass is placed on an empty wooden table by one hand", "glass placement and small reflections create the beat", "gentle static close-up", ["one clear water glass only", "plain empty wooden table", "hand"], common_avoid, "make the reset physical and specific"),
                BeatCandidate(3, "payoff", "breath_window_moment", "calm_breathing_open_window", "breath_by_window", "a calm person breathes beside an open bright window with relaxed posture", "body posture settles into calm focus", "medium side silhouette", ["open window", "curtains", "soft light", "plant"], common_avoid, "close with human payoff"),
            ],
            "classic but flexible light-to-action arc with visible change in each scene",
        ),
        (
            "tactile_first",
            "tactile_detail_hook",
            [
                BeatCandidate(1, "hook", "tactile_object_detail", "water_reflection_hook", "water_glass_closeup", "one clear water glass catches soft window reflections as a hand sets it on clean wood", "object contact and light reflection hook the viewer", "tight clean close-up", ["clear water glass", "light wood grain", "hand", "window reflection"], common_avoid, "start with tactile specificity"),
                BeatCandidate(2, "body_reset", "body_reset_gesture", "shoulder_posture_reset", "body_reset_medium", "a person rolls shoulders and turns toward soft morning light in a tidy room", "body gesture visibly shifts posture", "medium portrait side angle", ["relaxed posture", "plain clothing", "clean wall", "window light"], common_avoid, "avoid dead b-roll through clear movement"),
                BeatCandidate(3, "payoff", "sunlight_surface", "sunlight_surface_payoff", "sunlight_surface_detail", "soft sunlight moves across a clean surface beside simple fabric texture", "light settles into a calm final surface detail", "gentle static close-up", ["sunlight surface", "plain fabric", "clean wood", "soft shadow"], common_avoid, "end quietly without repeating the opening"),
            ],
            "object-first variant that avoids automatic curtain/window sequencing",
        ),
        (
            "motion_first",
            "body_reset_hook",
            [
                BeatCandidate(1, "hook", "body_reset_gesture", "body_motion_hook", "body_reset_medium", "a person steps into soft window light and performs one slow stretch", "human motion is readable immediately", "wide portrait frame with stable camera", ["person stretching", "window light", "plain wall", "tidy room"], common_avoid, "lead with human motion for stronger hook"),
                BeatCandidate(2, "environment_response", "fabric_texture", "curtain_fabric_response", "curtain_light_reveal", "plain curtain fabric moves gently as light changes across the room", "fabric and light respond to the movement", "soft handheld micro-motion", ["plain curtains", "fabric texture", "blank wall", "soft light"], common_avoid, "environment responds without text props"),
                BeatCandidate(3, "payoff", "before_after_micro_change", "clear_surface_after_reset", "surface_reset_detail", "a hand clears one small object from a clean surface, leaving a calmer simple composition", "before-after micro-change resolves the routine", "composed close side angle", ["clean surface", "single simple object", "hand", "soft daylight"], common_avoid, "close on visible before-after change"),
            ],
            "motion-first variant with no fixed water-glass middle beat",
        ),
    ]


def _fit_scene_count(beats: list[BeatCandidate], scene_count: int) -> list[BeatCandidate]:
    target = max(1, scene_count)
    if target <= len(beats):
        return [BeatCandidate(**{**beat.to_dict(), "scene_index": index}) for index, beat in enumerate(beats[:target], start=1)]
    fitted = list(beats)
    while len(fitted) < target:
        source = beats[-1]
        fitted.append(BeatCandidate(**{**source.to_dict(), "scene_index": len(fitted) + 1, "role": "payoff"}))
    return fitted


def _sanitize_visual_intent(idea: str, script: str) -> str:
    text = PromptBuilder._sanitize_visual_text(idea or "")
    for literal in _script_literals(script):
        if literal and literal.lower() in text.lower():
            text = text.replace(literal, " ")
    if not text or len(text.split()) < 4:
        text = "clean morning routine with soft light, tactile physical action, and calm human payoff"
    return " ".join(text.split()).strip(" ,.;")


def _script_literals(script: str) -> list[str]:
    import re

    return [part.strip(" .!?;:") for part in re.split(r"(?<=[.!?])\s+", script or "") if part.strip()]


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out
