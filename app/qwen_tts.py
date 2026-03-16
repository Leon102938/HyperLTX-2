import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

REPO_ROOT = Path("/workspace/Qwen3-TTS")
if REPO_ROOT.exists():
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

BASE_URL = ""
try:
    import os

    BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
except Exception:
    BASE_URL = ""

STATUS_DIR = Path("/workspace/status")
EXPORT_DIR = Path("/workspace/exports")
MODEL_PATH = Path("/workspace/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice")
ENV_FLAG = STATUS_DIR / "qwen_tts_env_ready"
TOKENIZER_FLAG = STATUS_DIR / "qwen_tts_tokenizer_ready"
MODEL_FLAG = STATUS_DIR / "qwen_tts_model_ready"
TOKENIZER_PATH = Path("/workspace/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz")

_MODEL = None
_MODEL_LOCK = threading.Lock()


class QwenCustomVoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("German", min_length=1)
    speaker: str = Field("Ryan", min_length=1)
    instruct: Optional[str] = Field(
        "A deep male German-speaking voice, slightly faster pace, a bit more energetic and lively, confident, warm, natural, realistic, premium YouTube voiceover style, clear pronunciation, engaging but not overexcited."
    )
    output_name: Optional[str] = None


def _ready_payload() -> dict:
    env_ready = ENV_FLAG.exists()
    tokenizer_ready = TOKENIZER_FLAG.exists() and TOKENIZER_PATH.exists()
    model_ready = MODEL_FLAG.exists() and MODEL_PATH.exists()
    return {
        "ready": env_ready and tokenizer_ready and model_ready,
        "env_ready": env_ready,
        "tokenizer_ready": tokenizer_ready,
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "source_repo": str(REPO_ROOT) if REPO_ROOT.exists() else None,
    }


def _sanitize_output_name(name: Optional[str]) -> str:
    if not name:
        return f"qwen_{uuid.uuid4().hex[:12]}.wav"

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    if not cleaned:
        cleaned = f"qwen_{uuid.uuid4().hex[:12]}"
    if not cleaned.endswith(".wav"):
        cleaned = f"{cleaned}.wav"
    return cleaned


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        from qwen_tts import Qwen3TTSModel

        kwargs = {
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        try:
            _MODEL = Qwen3TTSModel.from_pretrained(str(MODEL_PATH), **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            _MODEL = Qwen3TTSModel.from_pretrained(str(MODEL_PATH), **kwargs)

        return _MODEL


@router.get("/ready")
def qwen_tts_ready():
    payload = _ready_payload()
    payload["message"] = "Qwen TTS bereit." if payload["ready"] else "Qwen TTS nicht bereit."
    return payload


@router.get("/speakers")
def qwen_tts_speakers():
    ready = _ready_payload()
    if not ready["ready"]:
        raise HTTPException(status_code=503, detail=ready)

    try:
        model = _load_model()
        speakers = sorted(list(model.model.get_supported_speakers()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen speakers lookup failed: {exc}") from exc

    return {"speakers": speakers}


@router.get("/languages")
def qwen_tts_languages():
    ready = _ready_payload()
    if not ready["ready"]:
        raise HTTPException(status_code=503, detail=ready)

    try:
        model = _load_model()
        languages = sorted(list(model.model.get_supported_languages()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen languages lookup failed: {exc}") from exc

    return {"languages": languages}


@router.post("/custom_voice")
def qwen_tts_custom_voice(req: QwenCustomVoiceRequest):
    ready = _ready_payload()
    if not ready["ready"]:
        raise HTTPException(status_code=503, detail=ready)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = _sanitize_output_name(req.output_name)
    output_path = EXPORT_DIR / output_name

    try:
        model = _load_model()
        with torch.inference_mode():
            wavs, sample_rate = model.generate_custom_voice(
                text=req.text,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct or "",
            )
        sf.write(str(output_path), wavs[0], sample_rate, subtype="PCM_16")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen TTS generation failed: {exc}") from exc

    file_url = f"{BASE_URL}/exports/{output_name}" if BASE_URL else f"/exports/{output_name}"
    return {
        "ok": True,
        "speaker": req.speaker,
        "language": req.language,
        "sample_rate": sample_rate,
        "output_name": output_name,
        "output_path": str(output_path),
        "file_url": file_url,
    }
