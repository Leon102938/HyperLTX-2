from __future__ import annotations

from typing import Any

from agent_core.schemas import DirectorOutput, JobInput, SceneIntent, StyleLock, VariationDirective


class PromptBuilder:
    BUILDER_VERSION = "phaseA_scene_world_contract_v2"
    BASE_FORBIDDEN_VISUALS = [
        "readable text",
        "handwriting",
        "paper pages",
        "notebooks",
        "documents",
        "screens or UI facing camera",
        "labels",
        "logos",
        "posters",
        "signs",
        "generated subtitles inside the scene",
        "typography",
        "glyphs",
        "letters",
        "numbers",
    ]
    SOCIAL_FORBIDDEN_VISUALS = [
        "paper",
        "notebook",
        "document pages",
        "handwriting",
        "screens",
        "visible UI",
        "labels",
        "logos",
        "posters",
        "signs",
        "subtitles inside the generated scene",
        "typography",
        "glyphs",
        "letters",
        "numbers",
        "office desk paper drift",
    ]

    def build_global_prompt(self, job: JobInput, director_output: DirectorOutput) -> str:
        clauses = [
            director_output.creative_brief.concept,
            f"Hook: {director_output.creative_brief.hook}.",
            f"Style lock: {director_output.style_lock.visual_identity}, {director_output.style_lock.color_palette}.",
            f"Camera language: {director_output.style_lock.camera_language}.",
        ]
        if job.extra_llm_instruction:
            clauses.append(f"Extra instruction: {job.extra_llm_instruction}.")
        return self._join_clauses(clauses)

    def build_scene_prompt(
        self,
        *,
        job: JobInput,
        description: str,
        scene_text: str,
        scene_intent: SceneIntent,
        director_output: DirectorOutput,
    ) -> tuple[str, dict[str, Any]]:
        style_lock = director_output.style_lock
        scene_world_contract = self.build_scene_world_contract(
            job=job,
            description=description,
            scene_text=scene_text,
            scene_intent=scene_intent,
            director_output=director_output,
        )
        clauses = [
            f"WORLD / SETTING: {scene_world_contract['environment']}. Visible anchor: {scene_world_contract['visual_anchor']}.",
            f"SUBJECT / ACTION: {scene_world_contract['visible_subject']}; {scene_world_contract['action']}. One clear human or subject action only.",
            f"CAMERA / LIGHTING: {scene_world_contract['camera']}. Lighting: {scene_world_contract['lighting']}.",
            f"STYLE LOCK: {scene_world_contract['style_continuity']}. Palette: {style_lock.color_palette}. Texture: {style_lock.texture}.",
            f"ALLOWED VISUALS: {', '.join(scene_world_contract['allowed_props'])}.",
            f"FORBIDDEN VISUALS: {', '.join(scene_world_contract['forbidden_props'])}.",
            f"TEXT RISK POLICY: {scene_world_contract['text_risk_policy']}.",
        ]
        if scene_world_contract.get("social_format_rules"):
            clauses.append(f"SOCIAL FORMAT CONTRACT: {scene_world_contract['social_format_rules']}.")
        if scene_intent.opening_emphasis:
            clauses.append(f"OPENING EMPHASIS: {director_output.prompt_guidance.opening_shot}.")
        clauses.append(f"NARRATIVE ROLE: {scene_intent.narrative_role}.")
        if scene_text:
            clauses.append(f"STORY BEAT: {scene_text}.")
        if scene_intent.prompt_keywords:
            clauses.append(f"KEYWORDS: {', '.join(scene_intent.prompt_keywords[:5])}.")
        prompt_text = self._join_clauses(clauses)
        return prompt_text, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "scene",
            "director_mode": director_output.mode,
            "scene_world_contract": scene_world_contract,
            "opening_emphasis": scene_intent.opening_emphasis,
            "scene_role": scene_intent.narrative_role,
            "hook_focus": scene_intent.hook_focus,
            "visual_goal": scene_intent.visual_goal,
            "keywords": scene_intent.prompt_keywords,
            "style_keep": style_lock.keep,
            "style_avoid": style_lock.avoid,
        }

    def build_variation_prompt(
        self,
        *,
        scene_prompt_text: str,
        scene_intent: SceneIntent,
        style_lock: StyleLock,
        director_output: DirectorOutput,
        variation: VariationDirective | dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        variation_intent = self._get_value(variation, "intent")
        shot_type = self._get_value(variation, "shot_type")
        framing_hint = self._get_value(variation, "framing_hint")
        prompt_delta = self._get_value(variation, "prompt_delta")
        camera_style = self._get_value(variation, "camera_style")
        camera_motion = self._get_value(variation, "camera_motion")
        style_bias = self._get_value(variation, "style_bias")
        label = self._get_value(variation, "label")
        clauses = [
            scene_prompt_text,
            f"Variation intent: {variation_intent}.",
            f"Shot variation: {shot_type}.",
            f"Framing: {framing_hint}.",
            f"Prompt delta: {prompt_delta}.",
        ]
        if camera_style:
            clauses.append(f"Camera style: {camera_style}.")
        if camera_motion:
            clauses.append(f"Camera motion: {camera_motion}.")
        if style_bias:
            clauses.append(f"Style bias: {style_bias}.")
        if director_output.prompt_guidance.camera_cues:
            clauses.append(f"Camera cues: {', '.join(director_output.prompt_guidance.camera_cues[:3])}.")
        if style_lock.keep:
            clauses.append(f"Keep visual anchors: {', '.join(style_lock.keep[:2])}.")
        forbidden_visuals = self._forbidden_visuals(style_lock=style_lock, director_output=director_output)
        clauses.append(
            "World contract remains active: do not introduce new props, locations, screens, paper, labels, or readable text."
        )
        clauses.append(f"Forbidden visuals still apply: {', '.join(forbidden_visuals)}.")
        clauses.append("Variation is additive only: keep the same subject, setting, lighting logic, and text risk policy.")
        prompt_text = self._join_clauses(clauses)
        return prompt_text, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "variation",
            "director_mode": director_output.mode,
            "scene_role": scene_intent.narrative_role,
            "variation_label": label,
            "shot_type": shot_type,
            "creative_intent": variation_intent,
            "style_bias": style_bias,
            "forbidden_visuals": forbidden_visuals,
            "contract_preserved": True,
        }

    def build_storyboard_effective_prompt(
        self,
        *,
        scene_prompt_text: str,
        candidate_prompt_text: str,
        scene_world_contract: dict[str, Any] | None,
        variation: Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        contract = scene_world_contract or {}
        variation_intent = self._get_optional_value(variation, "creative_intent") or self._get_optional_value(
            variation, "intent"
        )
        shot_type = self._get_optional_value(variation, "shot_type")
        framing_hint = self._get_optional_value(variation, "framing_hint")
        prompt_delta = self._get_optional_value(variation, "prompt_delta")
        camera_style = self._get_optional_value(variation, "camera_style")
        camera_motion = self._get_optional_value(variation, "camera_motion")

        allowed_visuals = self._unique_terms(list(contract.get("allowed_props") or []), limit=8)
        forbidden_visuals = self._unique_terms(
            [
                *list(contract.get("forbidden_props") or []),
                "no readable text",
                "no handwriting",
                "no paper",
                "no notebook",
                "no document pages",
                "no screens or UI",
                "no labels",
                "no logos",
                "no posters",
                "no signs",
                "no typography",
                "no glyphs",
                "no letters",
                "no numbers",
            ],
            limit=32,
        )
        storyboard_negative = (
            "No readable text, no handwriting, no paper, no notebook, no document pages, "
            "no screens or UI facing camera, no labels, no logos, no posters, no signs, no typography, "
            "no glyphs, no letters, no numbers; use clean unlabeled surfaces only."
        )

        clauses = [
            f"Scene keyframe: {self._short_clause(contract.get('environment') or scene_prompt_text, 180)}.",
            f"Visible subject/action: {self._short_clause(contract.get('visible_subject'), 140)}; {self._short_clause(contract.get('action'), 140)}.",
            f"Variation: {self._short_clause(shot_type, 80)}; {self._short_clause(variation_intent, 140)}.",
            f"Framing: {self._short_clause(framing_hint, 120)}.",
            f"Camera/light: {self._short_clause(camera_style or contract.get('camera'), 140)}; {self._short_clause(camera_motion or contract.get('lighting'), 140)}.",
            f"Allowed visuals: {', '.join(allowed_visuals)}.",
            f"Forbidden visuals: {', '.join(forbidden_visuals)}.",
            f"Text risk policy: {contract.get('text_risk_policy') or storyboard_negative}.",
            "Single clean representative storyboard still, sharp composition, no motion blur, phone-readable subject silhouette.",
            storyboard_negative,
        ]
        if prompt_delta:
            clauses.insert(4, f"Controlled prompt delta: {self._short_clause(prompt_delta, 160)}.")
        if contract.get("social_format_rules"):
            clauses.append(f"Social format contract: {self._short_clause(contract['social_format_rules'], 220)}.")
        if candidate_prompt_text:
            clauses.append(f"Candidate prompt source: {self._short_clause(candidate_prompt_text, 260)}.")

        prompt_text = self._join_clauses(clauses)
        return prompt_text, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "storyboard_effective",
            "prompt_source": "scene_world_contract_candidate_variation",
            "contract_fields_used": [
                "visible_subject",
                "environment",
                "action",
                "allowed_props",
                "forbidden_props",
                "lighting",
                "camera",
                "text_risk_policy",
                "social_format_rules",
            ],
            "variation_intent": variation_intent,
            "shot_type": shot_type,
            "forbidden_visuals": forbidden_visuals,
            "contract_preserved": True,
            "social_tip_visual_guard": bool(contract.get("social_tip_visual_guard")),
        }

    def build_scene_world_contract(
        self,
        *,
        job: JobInput,
        description: str,
        scene_text: str,
        scene_intent: SceneIntent,
        director_output: DirectorOutput,
    ) -> dict[str, Any]:
        style_lock = director_output.style_lock
        social_guard = bool(director_output.metadata.get("social_tip_visual_guard"))
        allowed_props = self._allowed_visuals(
            scene_intent=scene_intent,
            style_lock=style_lock,
            director_output=director_output,
            social_guard=social_guard,
        )
        forbidden_props = self._forbidden_visuals(style_lock=style_lock, director_output=director_output)
        text_policy = (
            "No readable text, no handwriting, no document surfaces, no screens or UI toward camera, "
            "no labels, logos, posters, signs, typography, glyphs, letters, or numbers; use clean unlabeled surfaces."
        )
        social_rules = ""
        if social_guard:
            social_rules = (
                "portrait social tip must read at phone size; one clear human action per scene; "
                "avoid office, desk, paper, screen, and writing drift unless explicitly allowed; "
                "generated subtitles belong only to the external subtitle pass, never inside the scene image"
            )

        return {
            "visible_subject": self._short_clause(scene_intent.hook_focus or description),
            "visual_anchor": self._short_clause(scene_intent.visual_goal or description),
            "environment": self._short_clause(description or scene_text or scene_intent.visual_goal),
            "action": self._short_clause(scene_intent.shot_intent or scene_intent.hook_focus),
            "allowed_props": allowed_props,
            "forbidden_props": forbidden_props,
            "lighting": self._short_clause(style_lock.lighting),
            "camera": self._short_clause(style_lock.camera_language),
            "style_continuity": self._short_clause(
                f"{style_lock.visual_identity}; {', '.join(style_lock.keep[:3])}"
            ),
            "text_risk_policy": text_policy,
            "social_format_rules": social_rules,
            "source": "prompt_builder_v2",
            "social_tip_visual_guard": social_guard,
            "social_tip_visual_guard_family": director_output.metadata.get("social_tip_visual_guard_family"),
        }

    @staticmethod
    def _join_clauses(clauses: list[str]) -> str:
        seen: set[str] = set()
        cleaned: list[str] = []
        for clause in clauses:
            normalized = " ".join(str(clause).split()).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return " ".join(cleaned)

    @staticmethod
    def _get_value(payload: VariationDirective | dict[str, Any], key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key)

    @staticmethod
    def _get_optional_value(payload: Any | None, key: str) -> Any:
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    def _allowed_visuals(
        self,
        *,
        scene_intent: SceneIntent,
        style_lock: StyleLock,
        director_output: DirectorOutput,
        social_guard: bool,
    ) -> list[str]:
        values = [
            *style_lock.keep,
            *scene_intent.prompt_keywords,
            *director_output.prompt_guidance.visual_language,
        ]
        if social_guard:
            values.extend(
                [
                    "clean unlabeled surfaces",
                    "plain everyday props",
                    "window light",
                    "human movement",
                    "hidden device faces",
                ]
            )
        return self._unique_terms(values, limit=10) or ["clear subject", "clean environment", "controlled props"]

    def _forbidden_visuals(self, *, style_lock: StyleLock, director_output: DirectorOutput) -> list[str]:
        values = [
            *self.BASE_FORBIDDEN_VISUALS,
            *style_lock.avoid,
            *director_output.prompt_guidance.negative_cues,
        ]
        if director_output.metadata.get("social_tip_visual_guard"):
            values.extend(self.SOCIAL_FORBIDDEN_VISUALS)
        return self._unique_terms(values, limit=60)

    @staticmethod
    def _unique_terms(values: list[Any], *, limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(str(value).split()).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _short_clause(value: Any, limit: int = 220) -> str:
        if value is None:
            return ""
        normalized = " ".join(str(value).split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip(" ,.;") + "."
