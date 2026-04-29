from __future__ import annotations

import re
from typing import Any

from agent_core.llm_adapter import LocalOpenAICompatibleLLMAdapter
from agent_core.prompt_builder import PromptBuilder
from agent_core.schemas import (
    CreativeBrief,
    DirectorOutput,
    JobInput,
    PromptGuidance,
    SceneIntent,
    StyleLock,
    VariationDirective,
)
from agent_core.style_memory import StyleMemory


class DirectorEngine:
    def __init__(
        self,
        *,
        llm_adapter: LocalOpenAICompatibleLLMAdapter | None = None,
        style_memory: StyleMemory | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter or LocalOpenAICompatibleLLMAdapter()
        self.style_memory = style_memory or StyleMemory()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def build_direction(self, *, job: JobInput, scene_beats: list[dict[str, Any]]) -> DirectorOutput:
        style_lock = self.style_memory.build_style_lock(job)
        fallback_output = self._build_rule_based_output(job=job, scene_beats=scene_beats, style_lock=style_lock)
        llm_result = self.llm_adapter.plan_director(job=job, scene_beats=scene_beats, fallback_output=fallback_output)
        if not llm_result.get("ok"):
            fallback_output.fallback_reason = str(llm_result.get("reason") or "director_llm_unavailable")
            fallback_output.llm_active = False
            fallback_output.llm_provider = str(llm_result.get("provider") or "") or None
            fallback_output.llm_model = str(llm_result.get("model") or "") or None
            fallback_output.llm_endpoint = str(llm_result.get("endpoint") or "") or None
            fallback_output.metadata["llm_status"] = "fallback"
            if fallback_output.llm_endpoint:
                fallback_output.metadata["llm_endpoint"] = fallback_output.llm_endpoint
            return fallback_output

        try:
            normalized = self._normalize_llm_payload(
                payload=llm_result["payload"],
                job=job,
                scene_beats=scene_beats,
                base_style_lock=style_lock,
            )
            normalized.mode = "llm_augmented"
            normalized.fallback_reason = None
            normalized.llm_active = True
            normalized.llm_provider = str(llm_result.get("provider") or "local_openai_compatible")
            normalized.llm_model = str(llm_result.get("model") or "")
            normalized.llm_endpoint = str(llm_result.get("endpoint") or "") or None
            normalized.metadata["llm_status"] = "used"
            if normalized.llm_endpoint:
                normalized.metadata["llm_endpoint"] = normalized.llm_endpoint
            return normalized
        except Exception as exc:
            fallback_output.fallback_reason = f"director_llm_payload_invalid: {exc}"
            fallback_output.llm_active = False
            fallback_output.llm_provider = str(llm_result.get("provider") or "") or None
            fallback_output.llm_model = str(llm_result.get("model") or "") or None
            fallback_output.llm_endpoint = str(llm_result.get("endpoint") or "") or None
            fallback_output.metadata["llm_status"] = "fallback_after_invalid_payload"
            if fallback_output.llm_endpoint:
                fallback_output.metadata["llm_endpoint"] = fallback_output.llm_endpoint
            return fallback_output

    def _build_rule_based_output(
        self,
        *,
        job: JobInput,
        scene_beats: list[dict[str, Any]],
        style_lock: StyleLock,
    ) -> DirectorOutput:
        opening_scene = scene_beats[0] if scene_beats else {"description": job.idea or job.script or "opening beat"}
        final_scene = scene_beats[-1] if scene_beats else opening_scene
        scene_intents = [
            self._build_scene_intent(job=job, scene_beat=scene_beat, total_scene_count=len(scene_beats))
            for scene_beat in scene_beats
        ]

        return DirectorOutput(
            mode="rule_based_fallback",
            active=True,
            fallback_reason=None,
            llm_active=False,
            creative_brief=CreativeBrief(
                concept=self._short_text(job.idea or job.script or "A cinematic short-form sequence"),
                hook=f"Start with {self._short_text(opening_scene.get('description') or opening_scene.get('scene_text') or 'a strong opening reveal')}",
                audience_intent="deliver a stronger, fast-readable cinematic beat with a clear hook",
                narrative_arc=self._narrative_arc(scene_beats),
                emotional_arc="curiosity to momentum to payoff",
                payoff=f"Resolve on {self._short_text(final_scene.get('description') or final_scene.get('scene_text') or 'a clear final image')}",
                notes=["Rule-based director fallback kept the existing planner compatible."],
            ),
            style_lock=style_lock,
            prompt_guidance=PromptGuidance(
                opening_shot=f"wide reveal of {self._short_text(opening_scene.get('description') or 'the core subject')}",
                visual_language=[
                    "clear silhouette",
                    "cohesive lighting logic",
                    "specific environmental detail",
                ],
                camera_cues=[
                    "strong opening frame",
                    "deliberate lens choice",
                    "avoid random motion",
                ],
                prompt_rules=[
                    "lead with visual intent before camera jargon",
                    "keep prompts concise and production-specific",
                    "preserve style lock across all scenes",
                ],
                negative_cues=["generic cinematic scene", "overwritten prompt walls", "floating camera instructions"],
            ),
            scene_intents=scene_intents,
            character_notes=self._character_notes(job),
            voice_notes=self._voice_notes(job),
            world_notes=["Keep one coherent world logic across every scene transition."],
            metadata={"scene_count": len(scene_beats)},
        )

    def _build_scene_intent(self, *, job: JobInput, scene_beat: dict[str, Any], total_scene_count: int) -> SceneIntent:
        scene_index = int(scene_beat["scene_index"])
        scene_id = str(scene_beat["scene_id"])
        scene_text = str(scene_beat.get("scene_text") or "")
        role = self._scene_role(scene_index, total_scene_count)
        keywords = self._keywords(scene_text or str(scene_beat.get("description") or ""))
        variation_directives = self._variation_directives_for_scene(role=role, scene_text=scene_text)
        return SceneIntent(
            scene_id=scene_id,
            scene_index=scene_index,
            narrative_role=role,
            hook_focus=self._hook_focus(role, scene_text),
            emotional_beat=self._emotional_beat(role),
            visual_goal=self._visual_goal(role, scene_text),
            shot_intent=self._shot_intent(role, scene_text),
            opening_emphasis=scene_index == 1,
            transition_note=self._transition_note(scene_index, total_scene_count),
            prompt_keywords=keywords,
            variation_directives=variation_directives,
            notes=[f"Rule-based scene intent for {scene_id}."],
        )

    def _normalize_llm_payload(
        self,
        *,
        payload: dict[str, Any],
        job: JobInput,
        scene_beats: list[dict[str, Any]],
        base_style_lock: StyleLock,
    ) -> DirectorOutput:
        normalized_payload = dict(payload)
        if not self._looks_like_director_output_payload(normalized_payload):
            normalized_payload = self._coerce_scene_map_payload(
                payload=normalized_payload,
                job=job,
                scene_beats=scene_beats,
                base_style_lock=base_style_lock,
            )
        normalized_payload["style_lock"] = self.style_memory.build_style_lock(
            job,
            override=normalized_payload.get("style_lock"),
        ).model_dump(mode="json")

        by_index = {int(item["scene_index"]): item for item in scene_beats}
        scene_intents: list[dict[str, Any]] = []
        for index, scene in enumerate(normalized_payload.get("scene_intents") or [], start=1):
            beat = by_index.get(int(scene.get("scene_index", index)), scene_beats[index - 1])
            scene_payload = dict(scene)
            scene_payload["scene_id"] = str(beat["scene_id"])
            scene_payload["scene_index"] = int(beat["scene_index"])
            scene_intents.append(scene_payload)
        normalized_payload["scene_intents"] = scene_intents

        output = DirectorOutput.model_validate(normalized_payload)
        if len(output.scene_intents) != len(scene_beats):
            raise ValueError("scene_intent_count_mismatch")
        return output

    @staticmethod
    def _looks_like_director_output_payload(payload: dict[str, Any]) -> bool:
        required_keys = {"mode", "creative_brief", "prompt_guidance", "scene_intents"}
        return required_keys.issubset(payload.keys())

    def _coerce_scene_map_payload(
        self,
        *,
        payload: dict[str, Any],
        job: JobInput,
        scene_beats: list[dict[str, Any]],
        base_style_lock: StyleLock,
    ) -> dict[str, Any]:
        payload = self._unwrap_scene_map_payload(payload, scene_beats)
        scene_records: list[dict[str, Any]] = []
        visual_language: list[str] = []
        camera_cues: list[str] = []

        for scene_beat in scene_beats:
            scene_id = str(scene_beat["scene_id"])
            loose_scene = payload.get(scene_id)
            if not isinstance(loose_scene, dict):
                continue
            normalized_variations = self._scene_variations_from_loose_scene(loose_scene)
            scene_records.append(
                {
                    "scene_id": scene_id,
                    "scene_index": int(scene_beat["scene_index"]),
                    "visual_concept": str(loose_scene.get("visual_concept") or ""),
                    "prompt_seed": str(loose_scene.get("prompt_seed") or loose_scene.get("prompt") or ""),
                    "camera_movement": str(
                        loose_scene.get("camera_movement") or loose_scene.get("camera_motion") or ""
                    ),
                    "lighting_design": str(loose_scene.get("lighting_design") or ""),
                    "color_grading": str(loose_scene.get("color_grading") or ""),
                    "variations": normalized_variations,
                }
            )
            for value in (
                loose_scene.get("visual_concept"),
                loose_scene.get("prompt_seed"),
                loose_scene.get("lighting_design"),
                loose_scene.get("color_grading"),
            ):
                if value:
                    visual_language.append(self._short_text(str(value), 96))
            for variation in normalized_variations:
                movement = variation.get("camera_movement")
                if movement:
                    camera_cues.append(self._short_text(str(movement), 64))

        if not scene_records:
            raise ValueError("llm_scene_map_missing_known_scene_ids")

        first_scene = scene_records[0]
        last_scene = scene_records[-1]
        first_variation_prompt = self._first_variation_prompt(first_scene)
        narrative_arc_source = [
            {
                "description": record["visual_concept"] or scene_beats[index]["description"],
            }
            for index, record in enumerate(scene_records)
        ]

        scene_intents = [
            self._coerce_scene_intent_from_loose_payload(
                job=job,
                scene_beat=scene_beat,
                total_scene_count=len(scene_beats),
                loose_scene=next(
                    (record for record in scene_records if record["scene_id"] == str(scene_beat["scene_id"])),
                    None,
                ),
            )
            for scene_beat in scene_beats
        ]

        return {
            "mode": "llm_augmented",
            "active": True,
            "fallback_reason": None,
            "creative_brief": {
                "concept": self._short_text(job.idea or first_scene["visual_concept"] or job.script),
                "hook": self._short_text(first_variation_prompt or first_scene["visual_concept"] or job.idea),
                "audience_intent": "deliver a stronger cinematic hook, cleaner scene logic, and tighter visual intent",
                "narrative_arc": self._narrative_arc(narrative_arc_source),
                "emotional_arc": "intrigue to control to payoff",
                "payoff": self._short_text(last_scene["visual_concept"] or job.script),
                "notes": ["Normalized from local director LLM scene-map output."],
            },
            "style_lock": {
                "style_label": job.style,
                "visual_identity": self._short_text(first_scene["visual_concept"] or base_style_lock.visual_identity, 120),
                "color_palette": self._short_text(first_scene["color_grading"] or base_style_lock.color_palette, 120),
                "lighting": self._short_text(first_scene["lighting_design"] or base_style_lock.lighting, 120),
                "camera_language": base_style_lock.camera_language,
                "texture": base_style_lock.texture,
                "pacing": base_style_lock.pacing,
                "keep": list(base_style_lock.keep),
                "avoid": list(base_style_lock.avoid),
            },
            "prompt_guidance": {
                "opening_shot": self._short_text(first_variation_prompt or first_scene["visual_concept"], 120),
                "visual_language": self._unique_texts(visual_language, limit=4),
                "camera_cues": self._unique_texts(camera_cues, limit=4),
                "prompt_rules": [
                    "preserve the style lock across every variation",
                    "favor concise visual language over generic buzzwords",
                    "keep the opening shot immediately legible",
                ],
                "negative_cues": list(base_style_lock.avoid[:3]),
            },
            "scene_intents": scene_intents,
            "character_notes": self._character_notes(job),
            "voice_notes": self._voice_notes(job),
            "world_notes": ["Keep the same product-world logic and lighting continuity across all scenes."],
            "metadata": {
                "scene_count": len(scene_beats),
                "llm_payload_shape": "scene_map",
            },
        }

    @staticmethod
    def _unwrap_scene_map_payload(payload: dict[str, Any], scene_beats: list[dict[str, Any]]) -> dict[str, Any]:
        expected_scene_ids = {str(scene_beat["scene_id"]) for scene_beat in scene_beats}
        if expected_scene_ids.intersection(payload.keys()):
            return payload

        for key in ("scene_map", "scenes", "scene_data"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                if expected_scene_ids.intersection(candidate.keys()):
                    return candidate
            if isinstance(candidate, list):
                mapped: dict[str, Any] = {}
                for item in candidate:
                    if not isinstance(item, dict):
                        continue
                    scene_id = str(item.get("scene_id") or item.get("id") or "").strip()
                    if scene_id:
                        mapped[scene_id] = item
                if expected_scene_ids.intersection(mapped.keys()):
                    return mapped

        return payload

    @staticmethod
    def _scene_variations_from_loose_scene(loose_scene: dict[str, Any]) -> list[dict[str, Any]]:
        variations = loose_scene.get("variations")
        if isinstance(variations, list) and variations:
            return [variation for variation in variations if isinstance(variation, dict)]
        return []

    def _coerce_scene_intent_from_loose_payload(
        self,
        *,
        job: JobInput,
        scene_beat: dict[str, Any],
        total_scene_count: int,
        loose_scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        base_intent = self._build_scene_intent(job=job, scene_beat=scene_beat, total_scene_count=total_scene_count)
        if not loose_scene:
            return base_intent.model_dump(mode="json")

        variations = loose_scene.get("variations") or []
        variation_directives = self._coerce_variation_directives(
            variations=variations,
            fallback_directives=base_intent.variation_directives,
        )
        first_prompt = self._first_variation_prompt(loose_scene)
        first_camera = self._first_variation_camera(loose_scene)
        visual_concept = str(loose_scene.get("visual_concept") or "")
        lighting_design = str(loose_scene.get("lighting_design") or "")
        color_grading = str(loose_scene.get("color_grading") or "")

        payload = base_intent.model_dump(mode="json")
        payload["hook_focus"] = self._short_text(first_prompt or visual_concept or payload["hook_focus"], 140)
        payload["visual_goal"] = self._short_text(
            " ".join(part for part in [visual_concept, lighting_design, color_grading] if part),
            180,
        ) or payload["visual_goal"]
        payload["shot_intent"] = self._short_text(
            " ".join(part for part in [first_camera, first_prompt] if part),
            180,
        ) or payload["shot_intent"]
        payload["prompt_keywords"] = self._keywords(" ".join([visual_concept, lighting_design, color_grading]))
        payload["variation_directives"] = [directive.model_dump(mode="json") for directive in variation_directives]
        payload["notes"] = ["Normalized from local director LLM scene-map output."]
        return payload

    def _coerce_variation_directives(
        self,
        *,
        variations: list[dict[str, Any]],
        fallback_directives: list[VariationDirective],
    ) -> list[VariationDirective]:
        fallback_by_label = {directive.label: directive for directive in fallback_directives}
        directives: list[VariationDirective] = []
        for index, variation in enumerate(variations[:4], start=1):
            label = str(variation.get("variation_id") or variation.get("label") or f"variation_{index}")
            fallback = fallback_by_label.get(label)
            shot_type = fallback.shot_type if fallback else self._variation_shot_type(label)
            camera_style = str(variation.get("camera_style") or (fallback.camera_style if fallback else "")).strip() or None
            camera_motion = str(
                variation.get("camera_movement") or variation.get("camera_motion") or (fallback.camera_motion if fallback else "")
            ).strip() or None
            if not camera_style and not camera_motion:
                camera_motion = self._variation_default_camera_motion(shot_type)
            directives.append(
                VariationDirective(
                    label=label,
                    shot_type=shot_type,
                    intent=(fallback.intent if fallback else self._variation_intent(label)),
                    camera_style=camera_style,
                    camera_motion=camera_motion,
                    framing_hint=fallback.framing_hint if fallback else self._variation_framing_hint(shot_type),
                    prompt_delta=self._short_text(
                        str(variation.get("prompt") or variation.get("prompt_delta") or (fallback.prompt_delta if fallback else label)),
                        180,
                    ),
                    style_bias=fallback.style_bias if fallback else self._variation_style_bias(label),
                )
            )
        return directives or fallback_directives

    @staticmethod
    def _first_variation_prompt(loose_scene: dict[str, Any]) -> str:
        variations = loose_scene.get("variations") or []
        if not variations:
            return str(loose_scene.get("prompt_seed") or loose_scene.get("prompt") or "").strip()
        return str(variations[0].get("prompt") or "").strip()

    @staticmethod
    def _first_variation_camera(loose_scene: dict[str, Any]) -> str:
        variations = loose_scene.get("variations") or []
        if not variations:
            return str(loose_scene.get("camera_movement") or loose_scene.get("camera_motion") or "").strip()
        return str(variations[0].get("camera_movement") or variations[0].get("camera_motion") or "").strip()

    @staticmethod
    def _variation_shot_type(label: str) -> str:
        mapping = {
            "hook_master": "establishing",
            "kinetic_subject": "medium_action",
            "tactile_detail": "detail_closeup",
            "hero_resolve": "hero_tableau",
        }
        return mapping.get(label, "medium_action")

    @staticmethod
    def _variation_framing_hint(shot_type: str) -> str:
        mapping = {
            "establishing": "wide framing with one clear subject anchor",
            "medium_action": "medium framing with readable subject movement",
            "detail_closeup": "tight detail framing with tactile texture emphasis",
            "hero_tableau": "balanced hero framing with deliberate negative space",
        }
        return mapping.get(shot_type, "clear framing around the active subject")

    @staticmethod
    def _variation_style_bias(label: str) -> str | None:
        mapping = {
            "hook_master": "scale",
            "kinetic_subject": "motion",
            "tactile_detail": "texture",
            "hero_resolve": "clarity",
        }
        return mapping.get(label)

    @staticmethod
    def _variation_intent(label: str) -> str:
        mapping = {
            "hook_master": "show the world and subject relationship immediately",
            "kinetic_subject": "bring the viewer closer to the active subject beat",
            "tactile_detail": "surface one tactile detail without losing scene identity",
            "hero_resolve": "present the strongest composed key image for the beat",
        }
        return mapping.get(label, "strengthen the scene with a clearer visual variation")

    @staticmethod
    def _variation_default_camera_motion(shot_type: str) -> str:
        mapping = {
            "establishing": "slow push-in",
            "medium_action": "gentle lateral tracking",
            "detail_closeup": "subtle micro drift",
            "hero_tableau": "slow settle",
        }
        return mapping.get(shot_type, "restrained camera move")

    @staticmethod
    def _unique_texts(values: list[str], *, limit: int) -> list[str]:
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
    def _scene_role(scene_index: int, total_scene_count: int) -> str:
        if total_scene_count <= 1:
            return "single_scene_hook_payoff"
        if scene_index == 1:
            return "opening_hook"
        if scene_index == total_scene_count:
            return "final_payoff"
        return "escalation"

    @staticmethod
    def _transition_note(scene_index: int, total_scene_count: int) -> str | None:
        if total_scene_count <= 1:
            return None
        if scene_index == 1:
            return "handoff from hook into escalation"
        if scene_index == total_scene_count:
            return "resolve into a clean final image"
        return "push the narrative forward without losing visual continuity"

    @staticmethod
    def _hook_focus(role: str, scene_text: str) -> str:
        if role == "opening_hook":
            return f"immediate premise reveal: {DirectorEngine._short_text(scene_text or 'the core premise')}"
        if role == "final_payoff":
            return "reward the buildup with a clean, conclusive image"
        if role == "single_scene_hook_payoff":
            return "deliver premise and payoff in one strong beat"
        return f"raise momentum around {DirectorEngine._short_text(scene_text or 'the active scene beat')}"

    @staticmethod
    def _emotional_beat(role: str) -> str:
        mapping = {
            "opening_hook": "intrigue",
            "escalation": "momentum",
            "final_payoff": "satisfaction",
            "single_scene_hook_payoff": "curiosity into payoff",
        }
        return mapping.get(role, "focus")

    @staticmethod
    def _visual_goal(role: str, scene_text: str) -> str:
        if role == "opening_hook":
            return f"establish the world and subject fast through {DirectorEngine._short_text(scene_text or 'a bold opening image')}"
        if role == "final_payoff":
            return "land on the clearest, most resolved version of the subject"
        return f"make the beat legible and specific: {DirectorEngine._short_text(scene_text or 'the active scene beat')}"

    @staticmethod
    def _shot_intent(role: str, scene_text: str) -> str:
        if role == "opening_hook":
            return "start broad enough to read the world, then pull the eye toward the subject"
        if role == "final_payoff":
            return "resolve with a stronger hero image and calmer composition"
        if "detail" in scene_text.lower() or "interface" in scene_text.lower():
            return "use selective detail only when it strengthens readability"
        return "favor the most readable action-oriented composition for this beat"

    def _variation_directives_for_scene(self, *, role: str, scene_text: str) -> list[VariationDirective]:
        lowered = scene_text.lower()
        directives = [
            VariationDirective(
                label="hook_master",
                shot_type="establishing",
                intent="show the world and subject relationship immediately",
                camera_motion="slow push-in",
                framing_hint="wide environmental framing with one clear subject anchor",
                prompt_delta="favor geography, silhouette, and a stronger opening read",
                style_bias="scale",
            ),
            VariationDirective(
                label="kinetic_subject",
                shot_type="medium_action",
                intent="bring the viewer closer to the active subject beat",
                camera_motion="gentle lateral tracking",
                framing_hint="medium three-quarter framing with readable motion path",
                prompt_delta="favor subject focus, momentum, and cleaner directional movement",
                style_bias="motion",
            ),
            VariationDirective(
                label="tactile_detail",
                shot_type="detail_closeup",
                intent="surface one tactile detail without losing scene identity",
                camera_style="intimate cinematic close-up",
                framing_hint="tight detail framing with tactile texture emphasis",
                prompt_delta="favor tactile materials, clean unlabeled surfaces, and precise highlights without screens, paper, labels, or readable text",
                style_bias="texture",
            ),
            VariationDirective(
                label="hero_resolve",
                shot_type="hero_tableau",
                intent="present the strongest composed key image for the beat",
                camera_style="composed hero frame",
                framing_hint="balanced hero framing with deliberate negative space",
                prompt_delta="favor clarity, readable posture, and a stronger final image",
                style_bias="clarity",
            ),
        ]
        if role == "final_payoff" and any(token in lowered for token in {"final", "resolve", "complete", "clean"}):
            return [directives[3], directives[1], directives[0], directives[2]]
        if role == "escalation" and any(token in lowered for token in {"render", "progress", "move", "motion"}):
            return [directives[1], directives[2], directives[0], directives[3]]
        return directives

    @staticmethod
    def _narrative_arc(scene_beats: list[dict[str, Any]]) -> str:
        if len(scene_beats) <= 1:
            return "hook and payoff in one compressed cinematic beat"
        opening = DirectorEngine._short_text(scene_beats[0].get("description") or "")
        ending = DirectorEngine._short_text(scene_beats[-1].get("description") or "")
        return f"open on {opening}, escalate through the middle beats, and resolve on {ending}"

    @staticmethod
    def _keywords(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9_-]+", text.lower())
        stopwords = {"the", "and", "with", "from", "into", "this", "that", "scene", "shows", "show", "only"}
        keywords: list[str] = []
        for word in words:
            if len(word) < 4 or word in stopwords or word.isdigit():
                continue
            if word in keywords:
                continue
            keywords.append(word)
            if len(keywords) >= 6:
                break
        return keywords

    @staticmethod
    def _character_notes(job: JobInput) -> list[str]:
        source = f"{job.idea} {job.script}".strip()
        proper_nouns = []
        for match in re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", source):
            if match not in proper_nouns:
                proper_nouns.append(match)
        return [f"Keep {name} visually consistent if it appears on screen." for name in proper_nouns[:3]]

    @staticmethod
    def _voice_notes(job: JobInput) -> list[str]:
        if not job.use_voice:
            return []
        return [
            f"Voice anchor: {job.voice_id or 'default narrator'} with concise, confident delivery.",
            "Favor clear emphasis on the hook line and the final payoff phrase.",
        ]

    @staticmethod
    def _short_text(text: str, limit: int = 96) -> str:
        normalized = " ".join(str(text).split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
