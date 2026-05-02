from __future__ import annotations

from typing import Any

from agent_core.schemas import DirectorOutput, JobInput, SceneIntent, StyleLock, VariationDirective


class PromptBuilder:
    BUILDER_VERSION = "phaseA_scene_world_contract_v2"
    VISUAL_META_TERMS = (
        "social clip",
        "social-clip",
        "reel",
        "reels",
        "tiktok",
        "youtube",
        "post",
        "content",
        "content machine",
        "website",
        "webpage",
        "app",
        "ui",
        "interface",
        "screen",
        "phone",
        "smartphone",
        "mobile device",
        "feed",
        "browser",
        "dashboard",
    )
    DEVICE_UI_FORBIDDEN_VISUALS = [
        "phones",
        "smartphones",
        "mobile devices",
        "screens",
        "user interface",
        "app layout",
        "social media frame",
        "webpage",
        "website",
        "browser",
        "dashboard",
        "device surfaces",
        "split screen",
        "collage",
        "screenshot aesthetic",
    ]
    DEBUG_LABELS = (
        "WORLD / SETTING",
        "SUBJECT / ACTION",
        "FORBIDDEN VISUALS",
        "TEXT RISK POLICY",
        "STORY BEAT",
        "MOTIF SAFETY",
    )
    LEAKED_TERMS = (
        *DEBUG_LABELS,
        "Vorhang auf",
        "Stell ein Glas Wasser ab",
        "Atme ruhig am Fenster",
        "Morning Reset:",
        "social clip",
        "content",
        "website",
        "app",
        "ui",
        "screen",
        "phone",
        "smartphone",
    )
    POSITIVE_RISKY_TERMS = (
        "readable",
        "text-bearing",
        "typography",
        "letters",
        "numbers",
        "subtitles",
        "phone",
        "screen",
        "ui",
        "app",
        "website",
        "browser",
        "social",
        "content",
        "reel",
        "tiktok",
        "label",
        "logo",
    )
    POSITIVE_CONSTRAINT_TERMS = (
        "single full-frame shot",
        "one continuous scene",
        "plain empty wooden table",
        "one clear water glass only",
        "soft natural morning light",
        "blank wall",
        "clean physical space",
    )
    DEFAULT_BACKEND_PROMPT_POLICY = {
        "zimage": "positive_only",
        "ltx": "positive_plus_short_avoid",
    }
    LTX_SHORT_AVOID_TERMS = (
        "text",
        "logos",
        "phones",
        "screens",
        "user interface",
        "paper",
        "split screen",
        "collage",
        "panels",
        "black rectangle",
        "labels",
        "subtitles",
    )
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
            *DEVICE_UI_FORBIDDEN_VISUALS,
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
            *DEVICE_UI_FORBIDDEN_VISUALS,
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
            "MOTIF SAFETY: no phones, no screens, no user interface, no app layout, no social media frame, no webpage, no device surfaces, no black rectangle, no second object on the table, no paper, no labels, no overlaid text, no credits, no name labels, no typography, no split screen, no stacked panels, no collage, no embedded subtitles, no graphic layout.",
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
        visual_beat = self._visual_action_from_narration(scene_text)
        if visual_beat:
            clauses.append(f"VISUAL BEAT: {visual_beat}.")
        if scene_intent.prompt_keywords:
            clauses.append(f"KEYWORDS: {', '.join(scene_intent.prompt_keywords[:5])}.")
        debug_prompt = self._join_clauses(clauses)
        prompt_parts = self.compile_visual_prompt_parts(
            scene_world_contract=scene_world_contract,
            mode_id=str(director_output.metadata.get("creative_mode_id") or ""),
            style_id=str(director_output.metadata.get("creative_style_id") or ""),
        )
        model_prompt = prompt_parts["model_prompt"]
        scene_world_contract["model_prompt"] = model_prompt
        scene_world_contract["combined_model_prompt"] = prompt_parts["combined_model_prompt"]
        scene_world_contract["positive_model_prompt"] = prompt_parts["positive_model_prompt"]
        scene_world_contract["negative_model_prompt"] = prompt_parts["negative_model_prompt"]
        scene_world_contract["backend_prompt_policy"] = prompt_parts["backend_prompt_policy"]
        scene_world_contract["zimage_prompt_sent"] = prompt_parts["zimage_prompt_sent"]
        scene_world_contract["ltx_prompt_sent"] = prompt_parts["ltx_prompt_sent"]
        scene_world_contract["prompt_sent_to_backend_source"] = prompt_parts["prompt_sent_to_backend_source"]
        scene_world_contract["mode_id"] = director_output.metadata.get("creative_mode_id")
        scene_world_contract["style_id"] = director_output.metadata.get("creative_style_id")
        return debug_prompt, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "scene",
            "debug_prompt": debug_prompt,
            "model_prompt": model_prompt,
            "combined_model_prompt": prompt_parts["combined_model_prompt"],
            "positive_model_prompt": prompt_parts["positive_model_prompt"],
            "negative_model_prompt": prompt_parts["negative_model_prompt"],
            "backend_prompt_policy": prompt_parts["backend_prompt_policy"],
            "zimage_prompt_sent": prompt_parts["zimage_prompt_sent"],
            "ltx_prompt_sent": prompt_parts["ltx_prompt_sent"],
            "prompt_sent_to_backend_source": prompt_parts["prompt_sent_to_backend_source"],
            "prompt_audit": self.audit_model_prompt(model_prompt, script_text=job.script, extra_terms=[scene_text]),
            "mode_id": director_output.metadata.get("creative_mode_id"),
            "style_id": director_output.metadata.get("creative_style_id"),
            "director_mode": director_output.mode,
            "scene_world_contract": scene_world_contract,
            "opening_emphasis": scene_intent.opening_emphasis,
            "scene_role": scene_intent.narrative_role,
            "shot_recipe_id": scene_world_contract.get("shot_recipe_id"),
            "hook_function": scene_world_contract.get("hook_function"),
            "why_this_scene": scene_world_contract.get("why_this_scene"),
            "visual_energy_level": scene_world_contract.get("visual_energy_level"),
            "risk_notes": scene_world_contract.get("risk_notes"),
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
        scene_model_prompt = self._get_optional_value(variation, "scene_positive_model_prompt") or self._get_optional_value(
            variation,
            "scene_model_prompt",
        )
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
        debug_prompt = self._join_clauses(clauses)
        prompt_parts = self.compile_visual_prompt_parts(
            scene_world_contract=director_output.metadata.get("current_scene_world_contract"),
            base_model_prompt=scene_model_prompt,
            variation_hint="; ".join(str(item) for item in [framing_hint, prompt_delta, camera_motion or camera_style] if item),
            mode_id=str(director_output.metadata.get("creative_mode_id") or ""),
            style_id=str(director_output.metadata.get("creative_style_id") or ""),
        )
        model_prompt = prompt_parts["model_prompt"]
        return debug_prompt, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "variation",
            "debug_prompt": debug_prompt,
            "model_prompt": model_prompt,
            "combined_model_prompt": prompt_parts["combined_model_prompt"],
            "positive_model_prompt": prompt_parts["positive_model_prompt"],
            "negative_model_prompt": prompt_parts["negative_model_prompt"],
            "backend_prompt_policy": prompt_parts["backend_prompt_policy"],
            "zimage_prompt_sent": prompt_parts["zimage_prompt_sent"],
            "ltx_prompt_sent": prompt_parts["ltx_prompt_sent"],
            "prompt_sent_to_backend_source": prompt_parts["prompt_sent_to_backend_source"],
            "prompt_audit": self.audit_model_prompt(model_prompt),
            "mode_id": director_output.metadata.get("creative_mode_id"),
            "style_id": director_output.metadata.get("creative_style_id"),
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
                "no phones",
                "no smartphones",
                "no user interface",
                "no app layout",
                "no social media frame",
                "no webpage",
                "no device surfaces",
                "no black rectangle",
                "no split screen",
                "no stacked panels",
                "no collage",
                "no multi-panel layout",
                "no embedded subtitles",
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
            "no phones, no screens, no user interface, no app layout, no social media frame, no webpage, "
            "no device surfaces, no black rectangle, no split screen, no stacked panels, no collage, no multi-panel layout, "
            "no embedded subtitles, no labels, no logos, no posters, no signs, no typography, "
            "no glyphs, no letters, no numbers; use clean unlabeled surfaces only."
        )

        clauses = [
            f"Scene keyframe: {self._sanitize_visual_text(self._short_clause(contract.get('environment') or scene_prompt_text, 180))}.",
            f"Visible subject/action: {self._sanitize_visual_text(self._short_clause(contract.get('visible_subject'), 140))}; {self._sanitize_visual_text(self._short_clause(contract.get('action'), 140))}.",
            f"Variation: {self._short_clause(shot_type, 80)}; {self._short_clause(variation_intent, 140)}.",
            f"Framing: {self._short_clause(framing_hint, 120)}.",
            f"Camera/light: {self._short_clause(camera_style or contract.get('camera'), 140)}; {self._short_clause(camera_motion or contract.get('lighting'), 140)}.",
            f"Allowed visuals: {', '.join(self._sanitize_allowed_props(allowed_visuals))}.",
            f"Forbidden visuals: {', '.join(forbidden_visuals)}.",
            f"Text risk policy: {contract.get('text_risk_policy') or storyboard_negative}.",
            "Single full-frame shot, one continuous scene, sharp composition, no motion blur, clear subject silhouette.",
            storyboard_negative,
        ]
        if prompt_delta:
            clauses.insert(4, f"Controlled prompt delta: {self._short_clause(prompt_delta, 160)}.")
        if contract.get("social_format_rules"):
            clauses.append(f"Social format contract: {self._short_clause(contract['social_format_rules'], 220)}.")
        if candidate_prompt_text:
            clauses.append(f"Candidate prompt source: {self._sanitize_visual_text(self._short_clause(candidate_prompt_text, 260))}.")

        debug_prompt = self._join_clauses(clauses)
        prompt_parts = self.compile_visual_prompt_parts(
            scene_world_contract=contract,
            base_model_prompt=str(contract.get("positive_model_prompt") or contract.get("model_prompt") or ""),
            variation_hint="; ".join(str(item) for item in [framing_hint, prompt_delta, camera_motion or camera_style] if item),
            mode_id=str(contract.get("mode_id") or ""),
            style_id=str(contract.get("style_id") or ""),
        )
        model_prompt = prompt_parts["model_prompt"]
        return debug_prompt, {
            "builder_version": self.BUILDER_VERSION,
            "prompt_kind": "storyboard_effective",
            "debug_prompt": debug_prompt,
            "model_prompt": model_prompt,
            "combined_model_prompt": prompt_parts["combined_model_prompt"],
            "positive_model_prompt": prompt_parts["positive_model_prompt"],
            "negative_model_prompt": prompt_parts["negative_model_prompt"],
            "backend_prompt_policy": prompt_parts["backend_prompt_policy"],
            "zimage_prompt_sent": prompt_parts["zimage_prompt_sent"],
            "ltx_prompt_sent": prompt_parts["ltx_prompt_sent"],
            "prompt_sent_to_backend_source": prompt_parts["prompt_sent_to_backend_source"],
            "effective_model_prompt": prompt_parts["zimage_prompt_sent"],
            "prompt_audit": self.audit_model_prompt(model_prompt),
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

    def build_model_prompt(self, scene_world_contract: dict[str, Any], *, variation_hint: str | None = None) -> str:
        return self.compile_visual_prompt_for_model(
            scene_world_contract=scene_world_contract,
            variation_hint=variation_hint,
            mode_id=str(scene_world_contract.get("mode_id") or ""),
            style_id=str(scene_world_contract.get("style_id") or ""),
        )

    def build_debug_prompt(self, clauses: list[str]) -> str:
        return self._join_clauses(clauses)

    def compile_visual_prompt_parts(
        self,
        *,
        scene_world_contract: dict[str, Any] | None,
        base_model_prompt: str | None = None,
        variation_hint: str | None = None,
        mode_id: str | None = None,
        style_id: str | None = None,
    ) -> dict[str, Any]:
        contract = scene_world_contract or {}
        positive = self._build_positive_model_prompt(contract, base_model_prompt=base_model_prompt, variation_hint=variation_hint)
        negative_terms = self._build_negative_model_terms(contract)
        negative_prompt = ", ".join(negative_terms)
        backend_prompts = self._build_backend_prompt_payloads(contract, positive, negative_terms)
        combined = str(backend_prompts["ltx_prompt_sent"])
        combined = self._limit_words(self._strip_debug_and_leaked_terms(combined), 140)
        return {
            "positive_model_prompt": positive,
            "negative_model_prompt": negative_prompt,
            "negative_model_terms": negative_terms,
            "model_prompt": combined,
            "combined_model_prompt": combined,
            "backend_prompt_policy": backend_prompts["backend_prompt_policy"],
            "zimage_prompt_sent": backend_prompts["zimage_prompt_sent"],
            "ltx_prompt_sent": combined,
            "ltx_short_avoid_terms": backend_prompts["ltx_short_avoid_terms"],
            "prompt_sent_to_backend_source": backend_prompts["prompt_sent_to_backend_source"],
            "model_prompt_word_count": len(combined.split()),
            "positive_model_prompt_word_count": len(positive.split()),
            "negative_terms_count": len(negative_terms),
            "mode_id": mode_id,
            "style_id": style_id,
        }

    def compile_visual_prompt_for_model(
        self,
        *,
        scene_world_contract: dict[str, Any] | None,
        base_model_prompt: str | None = None,
        variation_hint: str | None = None,
        mode_id: str | None = None,
        style_id: str | None = None,
    ) -> str:
        return self.compile_visual_prompt_parts(
            scene_world_contract=scene_world_contract,
            base_model_prompt=base_model_prompt,
            variation_hint=variation_hint,
            mode_id=mode_id,
            style_id=style_id,
        )["model_prompt"]

    def _build_backend_prompt_payloads(
        self,
        contract: dict[str, Any],
        positive_model_prompt: str,
        negative_terms: list[str],
    ) -> dict[str, Any]:
        policy = self._backend_prompt_policy(contract)
        zimage_policy = str(policy.get("zimage") or "positive_only")
        ltx_policy = str(policy.get("ltx") or "positive_plus_short_avoid")
        ltx_short_terms = self._short_avoid_terms(negative_terms, limit=12)

        if zimage_policy == "positive_only":
            zimage_prompt = positive_model_prompt
            zimage_source = "positive_model_prompt"
        else:
            zimage_prompt = self._join_clauses([positive_model_prompt, f"Avoid: {', '.join(ltx_short_terms[:6])}"])
            zimage_source = "combined_model_prompt"

        if ltx_policy == "positive_plus_short_avoid" and ltx_short_terms:
            ltx_prompt = self._join_clauses([positive_model_prompt, f"Avoid: {', '.join(ltx_short_terms)}"])
            ltx_source = "combined_model_prompt"
        else:
            ltx_prompt = positive_model_prompt
            ltx_source = "positive_model_prompt"

        return {
            "backend_prompt_policy": policy,
            "zimage_prompt_sent": self._limit_words(self._strip_debug_and_leaked_terms(zimage_prompt), 100),
            "ltx_prompt_sent": self._limit_words(self._strip_debug_and_leaked_terms(ltx_prompt), 140),
            "ltx_short_avoid_terms": ltx_short_terms,
            "prompt_sent_to_backend_source": {
                "zimage": zimage_source,
                "ltx": ltx_source,
            },
        }

    def _backend_prompt_policy(self, contract: dict[str, Any]) -> dict[str, str]:
        explicit = contract.get("backend_prompt_policy")
        if isinstance(explicit, dict):
            return {
                "zimage": str(explicit.get("zimage") or self.DEFAULT_BACKEND_PROMPT_POLICY["zimage"]),
                "ltx": str(explicit.get("ltx") or self.DEFAULT_BACKEND_PROMPT_POLICY["ltx"]),
            }
        return dict(self.DEFAULT_BACKEND_PROMPT_POLICY)

    def _short_avoid_terms(self, negative_terms: list[str], *, limit: int) -> list[str]:
        lowered_to_original = {term.lower(): term for term in negative_terms}
        ordered: list[str] = []
        for preferred in self.LTX_SHORT_AVOID_TERMS:
            if preferred in lowered_to_original:
                ordered.append(lowered_to_original[preferred])
                continue
            plural = f"{preferred}s"
            if plural in lowered_to_original:
                ordered.append(lowered_to_original[plural])
        for term in negative_terms:
            if len(ordered) >= limit:
                break
            if term not in ordered:
                ordered.append(term)
        return self._unique_terms(ordered, limit=limit)

    def _build_positive_model_prompt(
        self,
        contract: dict[str, Any],
        *,
        base_model_prompt: str | None = None,
        variation_hint: str | None = None,
    ) -> str:
        motif_id = str(contract.get("motif_id") or "")
        if motif_id == "curtain_opening_window_light":
            positive = (
                "A calm person gently opens plain fabric curtains in soft natural morning light. "
                "The room is simple and quiet with a blank wall, tidy bedding, warm fabric texture, and one clear human action. "
                "Single full-frame lifestyle shot, soft handheld micro-motion, clean physical space."
            )
        elif motif_id == "water_glass_empty_table":
            positive = (
                "One clear water glass only rests on a plain empty wooden table in soft natural morning light. "
                "A single hand gently places the glass, with warm wood texture and simple clean composition. "
                "Close-up lifestyle shot, uncluttered surface, calm morning atmosphere."
            )
        elif motif_id == "calm_breathing_open_window":
            positive = (
                "A calm person stands beside an open window in soft natural morning light, breathing slowly with relaxed posture. "
                "Curtains move gently near a simple plant and clean wall. "
                "Single full-frame continuous lifestyle shot, quiet payoff moment, clean physical space."
            )
        elif base_model_prompt:
            positive = self._sanitize_visual_text(base_model_prompt)
        else:
            allowed = self._sanitize_allowed_props(list(contract.get("allowed_props") or []))[:4]
            positive = self._join_clauses(
                [
                    self._sanitize_visual_text(contract.get("visible_subject")),
                    self._sanitize_visual_text(contract.get("action")),
                    self._sanitize_visual_text(contract.get("environment")),
                    ", ".join(allowed),
                    "single full-frame lifestyle shot",
                    "clean physical space",
                ]
            )
        if variation_hint:
            safe_hint = self._remove_positive_risky_terms(self._sanitize_visual_text(variation_hint))
            if safe_hint and not any(term in safe_hint.lower() for term in ("avoid", " no ", "without")):
                positive = self._join_clauses([positive, safe_hint])
        positive = self._remove_positive_risky_terms(self._strip_debug_and_leaked_terms(positive))
        return self._limit_words(positive, 100)

    def _build_negative_model_terms(self, contract: dict[str, Any]) -> list[str]:
        motif_id = str(contract.get("motif_id") or "")
        if motif_id == "water_glass_empty_table":
            terms = [
                "phones",
                "screens",
                "black rectangle",
                "second object",
                "paper",
                "labels",
                "logos",
                "text",
                "letters",
                "numbers",
                "subtitles",
                "user interface",
                "app layout",
                "website",
                "split screen",
                "collage",
            ]
        elif motif_id == "calm_breathing_open_window":
            terms = [
                "split screen",
                "collage",
                "panels",
                "subtitles",
                "text",
                "letters",
                "numbers",
                "phones",
                "screens",
                "user interface",
                "app layout",
                "website",
                "paper",
                "logos",
                "labels",
            ]
        else:
            terms = [
                "text",
                "letters",
                "numbers",
                "subtitles",
                "logos",
                "labels",
                "phones",
                "screens",
                "user interface",
                "app layout",
                "website",
                "paper",
                "notebook",
                "split screen",
                "collage",
                "panels",
                "black rectangle",
            ]
        filtered = [
            term
            for term in terms
            if not any(term == constraint or constraint in term for constraint in self.POSITIVE_CONSTRAINT_TERMS)
        ]
        return self._unique_terms(filtered, limit=25)

    def _legacy_compile_visual_prompt_for_model(
        self,
        *,
        scene_world_contract: dict[str, Any] | None,
        base_model_prompt: str | None = None,
        variation_hint: str | None = None,
        mode_id: str | None = None,
        style_id: str | None = None,
    ) -> str:
        contract = scene_world_contract or {}
        allowed = self._sanitize_allowed_props(list(contract.get("allowed_props") or []))[:6]
        forbidden = self._unique_terms(
            [
                *list(contract.get("forbidden_props") or []),
                "no text",
                "no fake text",
                "no typography",
                "no letters",
                "no numbers",
                "no subtitles",
                "no phone",
                "no screen",
                "no UI",
                "no app layout",
                "no website",
                "no black rectangle",
                "no paper",
                "no logo",
                "no label",
                "no sign",
                "no split screen",
                "no stacked panels",
                "no collage",
            ],
            limit=60,
        )
        if base_model_prompt:
            positive = self._sanitize_visual_text(base_model_prompt)
        else:
            positive = self._join_clauses(
                [
                    self._sanitize_visual_text(contract.get("visible_subject")),
                    self._sanitize_visual_text(contract.get("action")),
                    self._sanitize_visual_text(contract.get("environment")),
                    ", ".join(allowed),
                    self._sanitize_visual_text(contract.get("lighting")),
                    "single full-frame physical scene",
                ]
            )
        if variation_hint:
            positive = self._join_clauses([positive, self._sanitize_visual_text(variation_hint)])
        no_phrases = [item if str(item).lower().startswith("no ") else f"no {item}" for item in forbidden]
        prompt = self._join_clauses([positive, ", ".join(no_phrases)])
        prompt = self._strip_debug_and_leaked_terms(prompt)
        return prompt

    @classmethod
    def audit_model_prompt(
        cls,
        model_prompt: str,
        *,
        script_text: str | None = None,
        extra_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        terms = list(cls.LEAKED_TERMS)
        if script_text:
            terms.extend([part.strip() for part in __import__("re").split(r"(?<=[.!?])\s+", script_text) if part.strip()])
        terms.extend(extra_terms or [])
        leaked = [term for term in terms if term and cls._term_is_positive_leak(model_prompt, term)]
        return {
            "leaked_terms_detected": leaked,
            "no_debug_labels_in_model_prompt": not any(label in model_prompt for label in cls.DEBUG_LABELS),
            "no_script_snippets_in_model_prompt": not leaked,
        }

    @staticmethod
    def _term_is_positive_leak(text: str, term: str) -> bool:
        import re

        if not term:
            return False
        for clause in re.split(r"[,.;]", text):
            lowered = clause.lower()
            if not re.search(rf"\b{re.escape(term.lower())}\b", lowered):
                continue
            if re.search(r"\b(no|without|forbidden|avoid|never)\b", lowered):
                continue
            return True
        return False

    @classmethod
    def _strip_debug_and_leaked_terms(cls, text: str) -> str:
        cleaned = str(text or "")
        for label in cls.DEBUG_LABELS:
            cleaned = cleaned.replace(label, " ")
        for phrase in ("Vorhang auf", "Stell ein Glas Wasser ab", "Atme ruhig am Fenster", "Morning Reset:"):
            cleaned = cleaned.replace(phrase, " ")
        cleaned = __import__("re").sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ,;")

    @classmethod
    def _remove_positive_risky_terms(cls, text: str) -> str:
        import re

        replacements = {
            "readable faces": "natural faces",
            "readable face": "natural face",
            "readable depth": "clear depth",
            "text-bearing props": "clean unlabeled physical props",
            "text-bearing": "clean unlabeled",
            "social tip": "portrait lifestyle short",
        }
        cleaned = str(text or "")
        for source, target in replacements.items():
            cleaned = re.sub(rf"\b{re.escape(source)}\b", target, cleaned, flags=re.IGNORECASE)
        for term in cls.POSITIVE_RISKY_TERMS:
            cleaned = re.sub(rf"\b{re.escape(term)}s?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace(" ,", ",").replace(" .", ".")
        return cleaned.strip(" ,.;")

    @staticmethod
    def _limit_words(text: str, limit: int) -> str:
        words = str(text or "").split()
        if len(words) <= limit:
            return " ".join(words)
        return " ".join(words[:limit]).rstrip(" ,.;") + "."

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

        mode = director_output.metadata.get("creative_mode") or {}
        scene_key = f"scene_{scene_intent.scene_index}"
        mode_scene = ((mode.get("scene_arc") or {}).get(scene_key) or {}) if isinstance(mode, dict) else {}
        backend_prompt_policy = mode.get("backend_prompt_policy") if isinstance(mode, dict) else None
        payload = {
            "visible_subject": self._sanitize_visual_text(self._short_clause(scene_intent.hook_focus or description)),
            "visual_anchor": self._sanitize_visual_text(self._short_clause(scene_intent.visual_goal or description)),
            "environment": self._sanitize_visual_text(
                self._short_clause(
                    scene_intent.visual_goal
                    if social_guard and scene_intent.visual_goal
                    else self._visual_action_from_narration(description or scene_text)
                    or description
                    or scene_intent.visual_goal
                )
            ),
            "action": self._sanitize_visual_text(self._short_clause(scene_intent.shot_intent or scene_intent.hook_focus)),
            "allowed_props": self._sanitize_allowed_props(allowed_props),
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
            "mode_id": director_output.metadata.get("creative_mode_id"),
            "style_id": director_output.metadata.get("creative_style_id"),
            "backend_prompt_policy": backend_prompt_policy or self.DEFAULT_BACKEND_PROMPT_POLICY,
            "anti_patterns_checked": list(mode.get("anti_patterns") or []) if isinstance(mode, dict) else [],
        }
        if mode_scene:
            payload["scene_role"] = mode_scene.get("role")
            payload["motif_id"] = mode_scene.get("motif")
            payload["shot_recipe_id"] = mode_scene.get("shot_recipe_id")
            payload["hook_function"] = mode_scene.get("hook_function")
            payload["why_this_scene"] = mode_scene.get("why_this_scene")
            payload["visual_energy_level"] = mode_scene.get("visual_energy_level")
            payload["risk_notes"] = [
                "check fake text, debug labels, narration text, devices, paper, and panel layouts before backend use"
            ]
            payload["visible_subject"] = self._sanitize_visual_text(mode_scene.get("visual_action"))
            payload["action"] = self._sanitize_visual_text(mode_scene.get("visual_action"))
            payload["allowed_props"] = self._sanitize_allowed_props(list(mode_scene.get("safe_props") or payload["allowed_props"]))
            payload["forbidden_props"] = self._unique_terms(
                [
                    *list(mode_scene.get("hard_rules") or []),
                    *list(mode_scene.get("forbidden") or []),
                    *list(mode.get("global_forbidden") or []),
                    *payload["forbidden_props"],
                ],
                limit=80,
            )
        return payload

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
                    "clear glass",
                    "plain fabric",
                ]
            )
        return self._sanitize_allowed_props(self._unique_terms(values, limit=10)) or ["clear subject", "clean environment", "controlled props"]

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

    @classmethod
    def _sanitize_allowed_props(cls, values: list[Any]) -> list[str]:
        return [
            cls._sanitize_visual_text(normalized)
            for value in values
            if (normalized := " ".join(str(value).split()).strip())
            and not cls._contains_visual_meta_term(normalized)
        ]

    @classmethod
    def _sanitize_visual_text(cls, value: Any) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        text = cls._visual_action_from_narration(text) or text
        replacements = {
            "readable human action": "clear human action",
            "readable body movement": "clear body movement",
            "readable stretch": "clear stretch",
            "readable stand-up": "clear stand-up",
            "readable wake-up": "clear wake-up",
            "readable depth separation": "clear depth separation",
            "readable at a glance": "clear at a glance",
            "phone size": "small portrait frame",
            "no phone beside it": "empty surrounding surface",
            "no phone in hand": "empty hands",
            "no phones, no screens, no ui": "clean device-free composition",
            "no phone, no screen, no paper, no labels": "clean unlabeled tabletop",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
            text = text.replace(source.title(), target)
        text = cls._remove_visual_meta_terms(text)
        text = " ".join(text.split(" ,")).replace(" ,", ",")
        return text.strip(" ,.;") or "clean daily routine moment"

    @staticmethod
    def _visual_action_from_narration(value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        lowered = text.lower().strip(" .!?:;\"'")
        exact = {
            "vorhang auf": "A person gently opens plain fabric curtains in soft morning light",
            "vorhang öffnen": "A person gently opens plain fabric curtains in soft morning light",
            "vorhang oeffnen": "A person gently opens plain fabric curtains in soft morning light",
            "stell ein glas wasser ab": "One clear water glass only on a plain empty wooden table",
            "wasserglas abstellen": "One clear water glass only on a plain empty wooden table",
            "atme ruhig am fenster": "A calm person breathes beside an open window in soft light",
            "am fenster ruhig atmen": "A calm person breathes beside an open window in soft light",
        }
        if lowered in exact:
            return exact[lowered]
        if len(lowered) <= 90 and any(token in lowered for token in ("vorhang", "curtain")) and any(
            token in lowered for token in ("auf", "öffnen", "oeffnen", "open")
        ):
            return "A person gently opens plain fabric curtains in soft morning light"
        if any(token in lowered for token in ("wasserglas", "glas wasser", "water glass")):
            return "One clear water glass only on a plain empty wooden table"
        if any(token in lowered for token in ("atme", "atmen", "breathe", "breathing")) and "fenster" in lowered:
            return "A calm person breathes beside an open window in soft light"
        return ""

    @classmethod
    def _remove_visual_meta_terms(cls, text: str) -> str:
        cleaned = text
        for term in sorted(cls.VISUAL_META_TERMS, key=len, reverse=True):
            cleaned = __import__("re").sub(rf"\b{__import__('re').escape(term)}s?\b", " ", cleaned, flags=__import__("re").IGNORECASE)
        cleaned = __import__("re").sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _contains_visual_meta_term(cls, text: str) -> bool:
        import re

        lowered = text.lower()
        return any(re.search(rf"\b{re.escape(term)}s?\b", lowered) for term in cls.VISUAL_META_TERMS)

    @staticmethod
    def _short_clause(value: Any, limit: int = 220) -> str:
        if value is None:
            return ""
        normalized = " ".join(str(value).split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip(" ,.;") + "."
