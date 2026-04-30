#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    raise SystemExit(exit_code)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(stripped[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        raise


def _normalize_status(value: Any) -> str:
    status = str(value or "needs_review").strip().lower()
    return status if status in {"passed", "needs_review", "rejected"} else "needs_review"


def main() -> None:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        _emit(
            {
                "provider": "qwen3_vl",
                "real_vlm_inference_used": False,
                "status": "needs_review",
                "postability_score": 0.5,
                "issues": [],
                "warnings": [f"invalid subprocess input: {exc}"],
                "problem_frames": [],
                "summary": "qwen3_vl subprocess input failed",
            },
            exit_code=1,
        )

    model_dir = Path(request.get("model_dir") or "/workspace/models/Qwen3-VL-4B-Instruct-FP8")
    frames = [frame for frame in request.get("frames") or [] if frame.get("path")]
    prompt = str(request.get("prompt") or "Review these video frames. Return strict JSON.")
    max_new_tokens = int(request.get("max_new_tokens") or 180)

    try:
        from PIL import Image as PILImage
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        device_preference = os.environ.get("VISION_REVIEW_DEVICE", "auto").strip().lower()
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": True}
        if device_preference == "cpu":
            model_kwargs["device_map"] = "cpu"
        elif device_preference in {"cuda", "auto"}:
            model_kwargs["device_map"] = "auto"

        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
        images = [PILImage.open(str(frame["path"])).convert("RGB") for frame in frames]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]
        try:
            payload = _extract_json_object(generated)
        except Exception as exc:
            _emit(
                {
                    "provider": "qwen3_vl",
                    "real_vlm_inference_used": True,
                    "status": "needs_review",
                    "postability_score": 0.5,
                    "issues": [],
                    "warnings": [f"qwen3_vl returned non-json review: {exc}"],
                    "problem_frames": [],
                    "summary": generated.strip()[:500] or "qwen3_vl returned no review text",
                }
            )
        _emit(
            {
                "provider": "qwen3_vl",
                "real_vlm_inference_used": True,
                "status": _normalize_status(payload.get("status") or payload.get("take_visual_review_status")),
                "postability_score": float(payload.get("postability_score", 0.5)),
                "issues": list(payload.get("issues") or []),
                "warnings": list(payload.get("warnings") or []),
                "problem_frames": list(payload.get("problem_frames") or []),
                "summary": str(payload.get("summary") or "qwen3_vl take visual review"),
            }
        )
    except Exception as exc:
        _emit(
            {
                "provider": "qwen3_vl",
                "real_vlm_inference_used": False,
                "status": "needs_review",
                "postability_score": 0.5,
                "issues": [],
                "warnings": [f"qwen3_vl subprocess inference failed: {exc}"],
                "problem_frames": [],
                "summary": "qwen3_vl subprocess failed",
            },
            exit_code=1,
        )


if __name__ == "__main__":
    main()
