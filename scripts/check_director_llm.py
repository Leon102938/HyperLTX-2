#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from urllib.request import Request, urlopen


def extract_json_object(content: str) -> dict:
    text = content.strip()
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
    raise ValueError("no_json_object_found")


def main() -> int:
    base_url = os.environ.get("DIRECTOR_LLM_BASE_URL", "http://127.0.0.1:8011").rstrip("/")
    model = os.environ.get("DIRECTOR_LLM_MODEL", "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf").strip()

    with urlopen(f"{base_url}/v1/models", timeout=20) as response:
        models_payload = json.loads(response.read().decode("utf-8"))

    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 96,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one JSON object and no prose.",
            },
            {
                "role": "user",
                "content": (
                    'Return {"director_llm_active": true, "provider": "llama_cpp_local", '
                    f'"model": "{model}", "status": "ok"}}'
                ),
            },
        ],
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        chat_payload = json.loads(response.read().decode("utf-8"))

    content = chat_payload["choices"][0]["message"]["content"]
    parsed = extract_json_object(content)
    summary = {
        "endpoint": f"{base_url}/v1/chat/completions",
        "provider": "llama_cpp_local",
        "model": model,
        "models_response_count": len(models_payload.get("data", [])),
        "chat_ok": parsed.get("status") == "ok",
        "parsed": parsed,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["chat_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
