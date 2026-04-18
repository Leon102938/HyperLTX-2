#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
DEFAULT_FILE = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
DEFAULT_ALT_REPO = "sharpcaterpillar/Qwen3.6-35B-A3B-GGUF"
DEFAULT_MODEL_DIR = Path("/workspace/models/director/qwen3.6-35b-a3b/gguf")


def download(repo_id: str, filename: str, target_dir: Path, token: str | None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=token or None,
        )
    )
    return resolved


def main() -> int:
    repo_id = os.environ.get("DIRECTOR_LLM_MODEL_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO
    filename = os.environ.get("DIRECTOR_LLM_MODEL_FILE", DEFAULT_FILE).strip() or DEFAULT_FILE
    target_dir = Path(os.environ.get("DIRECTOR_LLM_MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()
    target_path = Path(os.environ.get("DIRECTOR_LLM_MODEL_PATH", str(target_dir / filename))).expanduser()
    alt_repo_id = os.environ.get("DIRECTOR_LLM_ALT_MODEL_REPO", DEFAULT_ALT_REPO).strip()
    alt_filename = os.environ.get("DIRECTOR_LLM_ALT_MODEL_FILE", filename).strip() or filename
    token = os.environ.get("HF_TOKEN", "").strip() or None

    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[director-model] present: {target_path}")
        return 0

    attempts = [(repo_id, filename)]
    if alt_repo_id and (alt_repo_id, alt_filename) != (repo_id, filename):
        attempts.append((alt_repo_id, alt_filename))

    errors: list[str] = []
    for current_repo, current_filename in attempts:
        try:
            resolved = download(current_repo, current_filename, target_dir, token)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{current_repo}:{current_filename} -> {exc}")
            continue

        if resolved != target_path:
            resolved.replace(target_path)
        print(f"[director-model] downloaded: {target_path}")
        print(f"[director-model] source: {current_repo}:{current_filename}")
        return 0

    print("[director-model] ERROR: could not download requested GGUF.", file=sys.stderr)
    for error in errors:
        print(f"[director-model] {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
