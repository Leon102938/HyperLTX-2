#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
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


def load_env_defaults() -> None:
    for path in (
        Path("/workspace/config/director_llm.env"),
        Path("/workspace/config/director_llm.env.local"),
    ):
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def request_json(url: str, *, data: bytes | None, headers: dict[str, str], timeout: float, retries: int, retry_delay_sec: float) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(retry_delay_sec)
    raise RuntimeError(f"director_llm_request_failed: {last_error}")


def main() -> int:
    load_env_defaults()
    base_url = os.environ.get("DIRECTOR_LLM_BASE_URL", "http://127.0.0.1:8011").rstrip("/")
    model = (
        os.environ.get("DIRECTOR_LLM_MODEL")
        or os.environ.get("DIRECTOR_LLM_MODEL_FILE")
        or "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
    ).strip()
    models_timeout = float(os.environ.get("DIRECTOR_LLM_CHECK_MODELS_TIMEOUT_SEC", "20"))
    chat_timeout = float(os.environ.get("DIRECTOR_LLM_CHECK_CHAT_TIMEOUT_SEC", "120"))
    retries = int(os.environ.get("DIRECTOR_LLM_CHECK_RETRIES", "2"))
    retry_delay_sec = float(os.environ.get("DIRECTOR_LLM_CHECK_RETRY_DELAY_SEC", "3"))

    models_payload = request_json(
        f"{base_url}/v1/models",
        data=None,
        headers={"Connection": "close"},
        timeout=models_timeout,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
    )

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
                "content": json.dumps(
                    {
                        "director_llm_active": True,
                        "provider": "llama_cpp_local",
                        "model": model,
                        "status": "ok",
                    }
                ),
            },
        ],
    }
    chat_payload = request_json(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "close"},
        timeout=chat_timeout,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
    )

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
