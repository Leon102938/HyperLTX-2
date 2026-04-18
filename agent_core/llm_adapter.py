from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_core.schemas import DirectorOutput, JobInput


QWEN36_LLAMA_CPP_LOCAL_PROFILES = {
    "qwen36_35b_a3b_q4_k_m",
    "qwen36_35b_a3b_local",
    "qwen36_local",
    "qwen36_llama_cpp_local",
    "qwen3.6_35b_a3b_q4_k_m",
}
QWEN36_LLAMA_CPP_DEFAULT_BASE_URL = "http://127.0.0.1:8011"
QWEN36_LLAMA_CPP_DEFAULT_MODEL = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"


class LocalOpenAICompatibleLLMAdapter:
    def plan_director(
        self,
        *,
        job: JobInput,
        scene_beats: list[dict[str, Any]],
        fallback_output: DirectorOutput,
    ) -> dict[str, Any]:
        settings = self._resolve_settings(job)
        if not settings["enabled"]:
            return {
                "ok": False,
                "reason": settings["reason"],
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "endpoint": settings.get("url"),
            }

        if settings.get("provider") == "llama_cpp_local":
            prompt = self._build_local_scene_map_prompt(job=job, scene_beats=scene_beats, fallback_output=fallback_output)
            response_format = {"type": "json_object"}
        else:
            prompt = self._build_user_prompt(job=job, scene_beats=scene_beats, fallback_output=fallback_output)
            response_format = {
                "type": "json_schema",
                "schema": fallback_output.__class__.model_json_schema(),
            }
        payload = {
            "model": settings["model"],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if settings["api_key"]:
            headers["Authorization"] = f"Bearer {settings['api_key']}"

        request = Request(
            settings["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=settings["timeout_sec"]) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            return {
                "ok": False,
                "reason": f"director_llm_request_failed: {exc}",
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "endpoint": settings.get("url"),
            }
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "reason": f"director_llm_non_json_response: {exc}",
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "endpoint": settings.get("url"),
            }

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {
                "ok": False,
                "reason": "director_llm_response_missing_message_content",
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "endpoint": settings.get("url"),
            }

        try:
            parsed = self._extract_json_object(content)
        except ValueError as exc:
            return {
                "ok": False,
                "reason": f"director_llm_invalid_json_content: {exc}",
                "provider": settings.get("provider"),
                "model": settings.get("model"),
                "endpoint": settings.get("url"),
            }

        return {
            "ok": True,
            "provider": settings.get("provider") or "local_openai_compatible",
            "model": settings["model"],
            "endpoint": settings["url"],
            "payload": parsed,
            "raw": raw,
        }

    @staticmethod
    def _resolve_settings(job: JobInput) -> dict[str, Any]:
        config_sources = []
        if isinstance(job.metadata.get("director_llm"), dict):
            config_sources.append(job.metadata["director_llm"])
        if isinstance(job.backend_overrides.get("director_llm"), dict):
            config_sources.append(job.backend_overrides["director_llm"])

        merged: dict[str, Any] = {}
        for source in config_sources:
            merged.update(source)

        profile = str(
            merged.get("profile")
            or os.environ.get("DIRECTOR_LLM_PROFILE")
            or ""
        ).strip().lower()
        local_profile_selected = profile in QWEN36_LLAMA_CPP_LOCAL_PROFILES
        base_url = str(
            merged.get("base_url")
            or os.environ.get("DIRECTOR_LLM_BASE_URL")
            or (QWEN36_LLAMA_CPP_DEFAULT_BASE_URL if local_profile_selected else "")
            or ""
        ).strip()
        model = str(
            merged.get("model")
            or os.environ.get("DIRECTOR_LLM_MODEL")
            or (QWEN36_LLAMA_CPP_DEFAULT_MODEL if local_profile_selected else "")
            or ""
        ).strip()
        api_key = str(
            merged.get("api_key")
            or os.environ.get("DIRECTOR_LLM_API_KEY")
            or ""
        ).strip()
        endpoint_path = str(
            merged.get("path")
            or os.environ.get("DIRECTOR_LLM_PATH")
            or "/v1/chat/completions"
        ).strip()
        provider = str(
            merged.get("provider")
            or os.environ.get("DIRECTOR_LLM_PROVIDER")
            or ("llama_cpp_local" if local_profile_selected else "local_openai_compatible")
        ).strip()
        temperature = float(
            merged.get("temperature")
            or os.environ.get("DIRECTOR_LLM_TEMPERATURE")
            or (0.25 if local_profile_selected else 0.6)
        )
        timeout_sec = float(
            merged.get("timeout_sec")
            or os.environ.get("DIRECTOR_LLM_TIMEOUT_SEC")
            or (240.0 if local_profile_selected else 45.0)
        )
        max_tokens = int(
            merged.get("max_tokens")
            or os.environ.get("DIRECTOR_LLM_MAX_TOKENS")
            or (1800 if local_profile_selected else 1400)
        )
        enabled_flag = merged.get("enabled")

        if enabled_flag is False:
            return {
                "enabled": False,
                "reason": "director_llm_disabled",
                "provider": provider,
                "model": model or None,
                "url": base_url or None,
            }
        if not base_url or not model:
            return {
                "enabled": False,
                "reason": "director_llm_not_configured",
                "provider": provider,
                "model": model or None,
                "url": base_url or None,
            }

        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if endpoint_path.startswith("http://") or endpoint_path.startswith("https://"):
            url = endpoint_path
        else:
            url = f"{base_url}{endpoint_path if endpoint_path.startswith('/') else '/' + endpoint_path}"
        return {
            "enabled": True,
            "url": url,
            "model": model,
            "api_key": api_key,
            "provider": provider,
            "temperature": temperature,
            "timeout_sec": timeout_sec,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a film director brain for a local video pipeline. "
            "Return only valid JSON. Keep prompts concise, cinematic, specific, and production-minded. "
            "Do not mention JSON in the output."
        )

    @staticmethod
    def _build_user_prompt(*, job: JobInput, scene_beats: list[dict[str, Any]], fallback_output: DirectorOutput) -> str:
        return json.dumps(
            {
                "job": {
                    "idea": job.idea,
                    "script": job.script,
                    "style": job.style,
                    "use_voice": job.use_voice,
                    "orientation": job.orientation,
                    "resolution": job.resolution,
                    "extra_llm_instruction": job.extra_llm_instruction,
                },
                "scene_beats": scene_beats,
                "required_json_shape": fallback_output.model_dump(mode="json"),
                "instructions": [
                    "Keep the same top-level keys and nested structure.",
                    "Preserve scene_ids and scene_index values.",
                    "Prefer stronger hook/opening, cleaner style lock, and concrete scene/variation intent.",
                    "Avoid generic buzzwords and long prompt walls.",
                ],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _build_local_scene_map_prompt(*, job: JobInput, scene_beats: list[dict[str, Any]], fallback_output: DirectorOutput) -> str:
        compact_shape: dict[str, Any] = {}
        intents_by_scene = {intent.scene_id: intent for intent in fallback_output.scene_intents}
        for scene_beat in scene_beats:
            scene_id = str(scene_beat["scene_id"])
            intent = intents_by_scene.get(scene_id)
            variation_shape = []
            for directive in (intent.variation_directives if intent else []):
                variation_shape.append(
                    {
                        "variation_id": directive.label,
                        "prompt": f"{directive.intent}; {directive.prompt_delta}",
                        "camera_movement": directive.camera_motion or directive.camera_style or "restrained movement",
                    }
                )
            compact_shape[scene_id] = {
                "visual_concept": str(scene_beat.get("description") or scene_beat.get("scene_text") or ""),
                "lighting_design": fallback_output.style_lock.lighting,
                "color_grading": fallback_output.style_lock.color_palette,
                "variations": variation_shape,
            }

        return json.dumps(
            {
                "job": {
                    "idea": job.idea,
                    "script": job.script,
                    "style": job.style,
                    "orientation": job.orientation,
                    "resolution": job.resolution,
                    "extra_llm_instruction": job.extra_llm_instruction,
                },
                "scene_beats": scene_beats,
                "required_json_shape": compact_shape,
                "instructions": [
                    "Return only one JSON object.",
                    "Use every scene_id as a top-level key.",
                    "For each scene include visual_concept, lighting_design, color_grading, and variations.",
                    "Each variation needs variation_id, prompt, and camera_movement.",
                    "Keep variation prompts concise, cinematic, and production-minded.",
                ],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and part.endswith("}"):
                    text = part
                    break
                if "\n" in part:
                    candidate = part.split("\n", 1)[1].strip()
                    if candidate.startswith("{") and candidate.endswith("}"):
                        text = candidate
                        break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(text[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            raise
