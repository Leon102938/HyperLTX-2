from __future__ import annotations

from typing import Any

from agent_core.schemas import DirectorOutput, JobInput, SceneIntent, StyleLock, VariationDirective


class PromptBuilder:
    BUILDER_VERSION = "phase5a_director_v1"

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
        clauses = []
        if scene_intent.opening_emphasis:
            clauses.append(f"Opening shot: {director_output.prompt_guidance.opening_shot}.")
        clauses.extend(
            [
                description,
                f"Narrative role: {scene_intent.narrative_role}.",
                f"Hook focus: {scene_intent.hook_focus}.",
                f"Visual goal: {scene_intent.visual_goal}.",
                f"Shot intent: {scene_intent.shot_intent}.",
                f"Style lock: {style_lock.visual_identity}, {style_lock.color_palette}, {style_lock.lighting}.",
                f"Camera: {style_lock.camera_language}.",
            ]
        )
        if scene_text:
            clauses.append(f"Story beat: {scene_text}.")
        if scene_intent.prompt_keywords:
            clauses.append(f"Keywords: {', '.join(scene_intent.prompt_keywords[:5])}.")
        if style_lock.keep:
            clauses.append(f"Keep: {', '.join(style_lock.keep[:3])}.")
        if style_lock.avoid:
            clauses.append(f"Avoid: {', '.join(style_lock.avoid[:3])}.")
        prompt_text = self._join_clauses(clauses)
        return prompt_text, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "scene",
            "director_mode": director_output.mode,
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
