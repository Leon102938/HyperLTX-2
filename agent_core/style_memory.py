from __future__ import annotations

from typing import Any

from agent_core.schemas import JobInput, StyleLock


class StyleMemory:
    def build_style_lock(self, job: JobInput, override: dict[str, Any] | None = None) -> StyleLock:
        base = self._base_style_lock(job)
        if not override:
            return base

        payload = {
            "style_label": str(override.get("style_label") or base.style_label),
            "visual_identity": str(override.get("visual_identity") or base.visual_identity),
            "color_palette": str(override.get("color_palette") or base.color_palette),
            "lighting": str(override.get("lighting") or base.lighting),
            "camera_language": str(override.get("camera_language") or base.camera_language),
            "texture": str(override.get("texture") or base.texture),
            "pacing": str(override.get("pacing") or base.pacing),
            "keep": self._merge_terms(base.keep, override.get("keep")),
            "avoid": self._merge_terms(base.avoid, override.get("avoid")),
        }
        return StyleLock.model_validate(payload)

    def _base_style_lock(self, job: JobInput) -> StyleLock:
        style_text = (job.style or "cinematic").strip()
        lowered = style_text.lower()

        visual_identity = "cinematic realism with deliberate composition"
        color_palette = "steel-blue shadows with warm practical highlights"
        lighting = "shaped contrast, readable faces, controlled highlights"
        camera_language = "purposeful lensing, clear subject anchor, restrained movement"
        texture = "clean detail, tactile materials, grounded contrast"
        pacing = "brisk beats with a confident opening and clean payoff"
        keep = ["clear subject silhouette", "cohesive lighting logic", "readable depth separation"]
        avoid = ["generic stock imagery", "muddy lighting", "text overlay", "visual clutter"]

        if "noir" in lowered:
            color_palette = "inky blacks, hard white edges, sparse warm spill"
            lighting = "hard contrast, practical pools, deep falloff"
            texture = "grainy shadows, reflective surfaces, crisp highlights"
        elif "anime" in lowered:
            visual_identity = "stylized cinematic anime composition"
            color_palette = "bold color separation with clean accent tones"
            lighting = "graphic light shapes and readable edge lighting"
            texture = "clean surfaces, polished gradients, selective detail"
        elif "documentary" in lowered:
            visual_identity = "grounded documentary realism"
            color_palette = "naturalistic neutrals with subtle contrast"
            lighting = "available-light realism with soft shaping"
            camera_language = "observational camera, grounded framing, subtle handheld restraint"
            pacing = "measured observational beats with a direct payoff"
        elif "retro" in lowered or "vintage" in lowered:
            color_palette = "aged amber mids, muted cyans, soft rolloff"
            texture = "analog grain, lived-in surfaces, gentle halation"

        return StyleLock(
            style_label=style_text,
            visual_identity=visual_identity,
            color_palette=color_palette,
            lighting=lighting,
            camera_language=camera_language,
            texture=texture,
            pacing=pacing,
            keep=keep,
            avoid=avoid,
        )

    @staticmethod
    def _merge_terms(base_terms: list[str], override_terms: Any) -> list[str]:
        merged = list(base_terms)
        if isinstance(override_terms, list):
            merged.extend(str(item).strip() for item in override_terms if str(item).strip())
        elif override_terms not in (None, ""):
            merged.append(str(override_terms).strip())

        seen: set[str] = set()
        result: list[str] = []
        for item in merged:
            normalized = item.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(item)
        return result
