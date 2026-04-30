#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen3-VL-4B-Instruct-FP8"
TARGET_DIR = Path("/workspace/models/Qwen3-VL-4B-Instruct-FP8")


def fail(message: str) -> None:
    print(f"[qwen3-vl] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing or empty file: {path}")


def verify_model_dir(root: Path) -> None:
    require_file(root / "config.json")

    if not (root / "tokenizer.json").is_file() and not (root / "tokenizer_config.json").is_file():
        fail(f"missing tokenizer.json or tokenizer_config.json in {root}")

    require_file(root / "preprocessor_config.json")

    index_path = root / "model.safetensors.index.json"
    require_file(index_path)

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read safetensors index {index_path}: {exc}")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail(f"invalid or empty weight_map in {index_path}")

    missing = []
    for shard_name in sorted(set(weight_map.values())):
        shard_path = root / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            missing.append(str(shard_path))

    if missing:
        fail("missing indexed shard(s): " + ", ".join(missing))

    incomplete = sorted(root.rglob("*.incomplete"))
    if incomplete:
        fail("incomplete download file(s) present: " + ", ".join(str(path) for path in incomplete))


def main() -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[qwen3-vl] downloading/verifying {REPO_ID} -> {TARGET_DIR}")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        fail(f"snapshot_download failed: {exc}")

    verify_model_dir(TARGET_DIR)
    print(f"[qwen3-vl] OK: model verified at {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
