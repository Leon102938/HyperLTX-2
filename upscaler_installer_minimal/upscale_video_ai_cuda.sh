#!/usr/bin/env bash
set -euo pipefail

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
err() { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

usage() {
  cat <<'EOF'
Usage:
  bash upscale_video_ai_cuda.sh --in input.mp4 --out output.mp4 [--model x2|x4|anime] [--target 1080x1920|none] [--tile 256] [--keep-frames]

Defaults:
  --model x2
  --target none
  --tile 256
EOF
}

require_arg() {
  local key="$1"
  local val="${2:-}"
  [[ -n "$val" ]] || { err "$key needs a value"; exit 1; }
}

INPUT=""
OUTPUT=""
MODEL_OPT="x2"
TARGET="none"
TILE="256"
KEEP_FRAMES="0"

BASE_DIR="/workspace/tools/realesrgan_ai"
REPO_DIR="${BASE_DIR}/Real-ESRGAN"
VENV_DIR="${BASE_DIR}/venv"

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in) require_arg "$1" "${2:-}"; INPUT="$2"; shift 2 ;;
    --out) require_arg "$1" "${2:-}"; OUTPUT="$2"; shift 2 ;;
    --model) require_arg "$1" "${2:-}"; MODEL_OPT="$2"; shift 2 ;;
    --target) require_arg "$1" "${2:-}"; TARGET="$2"; shift 2 ;;
    --tile) require_arg "$1" "${2:-}"; TILE="$2"; shift 2 ;;
    --keep-frames) KEEP_FRAMES="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$INPUT" && -f "$INPUT" ]] || { err "Input missing: $INPUT"; exit 2; }
[[ -n "$OUTPUT" ]] || { err "--out required"; exit 2; }
[[ "$MODEL_OPT" =~ ^(x2|x4|anime)$ ]] || { err "--model must be x2|x4|anime"; exit 2; }
[[ "$TILE" =~ ^[0-9]+$ ]] || { err "--tile must be numeric"; exit 2; }
if [[ "$TARGET" != "none" ]]; then
  [[ "$TARGET" =~ ^[0-9]+x[0-9]+$ ]] || { err "--target must be WxH or none"; exit 2; }
fi

[[ -d "$REPO_DIR" ]] || { err "Repo not found: $REPO_DIR (run setup_ai_cuda.sh)"; exit 3; }
[[ -d "$VENV_DIR" ]] || { err "Venv not found: $VENV_DIR (run setup_ai_cuda.sh)"; exit 3; }
command -v ffmpeg >/dev/null 2>&1 || { err "ffmpeg missing"; exit 3; }
command -v ffprobe >/dev/null 2>&1 || { err "ffprobe missing"; exit 3; }

source "${VENV_DIR}/bin/activate"

python - <<'PY'
from pathlib import Path
import torchvision
transforms_dir = Path(torchvision.__file__).resolve().parent / "transforms"
shim = transforms_dir / "functional_tensor.py"
if not shim.exists():
    shim.write_text(
        "# Auto-generated compatibility shim for BasicSR on newer torchvision\n"
        "from .functional import *\n",
        encoding="utf-8",
    )
    print("Created torchvision shim:", shim)
PY

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: torch CUDA not available")
    sys.exit(9)
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

MODEL_NAME=""
OUTSCALE=""
case "$MODEL_OPT" in
  x2) MODEL_NAME="RealESRGAN_x2plus"; OUTSCALE="2" ;;
  x4) MODEL_NAME="RealESRGAN_x4plus"; OUTSCALE="2" ;; # use x4 model then downscale to 2x-equivalent target if desired
  anime) MODEL_NAME="RealESRGAN_x4plus_anime_6B"; OUTSCALE="2" ;;
esac

FPS_RAW="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$INPUT" || true)"
if [[ -z "$FPS_RAW" || "$FPS_RAW" == "0/0" || "$FPS_RAW" == "N/A" ]]; then
  FPS_RAW="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "$INPUT" || true)"
fi
FPS="$(awk -v r="$FPS_RAW" 'BEGIN{split(r,a,"/"); if (a[2]=="" || a[2]==0) {print r} else {printf "%.6f", a[1]/a[2]}}')"
[[ -n "$FPS" ]] || { err "Could not detect FPS"; exit 4; }

HAS_AUDIO="0"
if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "$INPUT" | grep -q audio; then
  HAS_AUDIO="1"
fi

WORKDIR="$(mktemp -d /tmp/realesrgan_ai_XXXXXX)"
FRAMES_IN="${WORKDIR}/frames_in"
FRAMES_UP="${WORKDIR}/frames_up"
mkdir -p "$FRAMES_IN" "$FRAMES_UP"

cleanup() {
  if [[ "$KEEP_FRAMES" == "1" ]]; then
    log "Keeping frames: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

log "Extracting frames..."
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" -vsync 0 "${FRAMES_IN}/frame_%08d.png"

log "Running Real-ESRGAN AI (PyTorch CUDA)..."
cd "$REPO_DIR"
python inference_realesrgan.py \
  -i "$FRAMES_IN" \
  -o "$FRAMES_UP" \
  -n "$MODEL_NAME" \
  --outscale "$OUTSCALE" \
  --tile "$TILE" \
  --suffix '' \
  --ext png

OUT_COUNT="$(find "$FRAMES_UP" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
[[ "$OUT_COUNT" -gt 0 ]] || { err "No upscaled frames generated"; exit 5; }

VF_ARGS=()
if [[ "$TARGET" != "none" ]]; then
  TW="${TARGET%x*}"
  TH="${TARGET#*x}"
  VF_ARGS=( -vf "scale=${TW}:${TH}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${TW}:${TH}:(ow-iw)/2:(oh-ih)/2:black" )
fi

ENCODER="libx264"
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
  ENCODER="h264_nvenc"
fi

log "Encoding output with $ENCODER..."
if [[ "$HAS_AUDIO" == "1" ]]; then
  if [[ "$ENCODER" == "h264_nvenc" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -framerate "$FPS" -i "${FRAMES_UP}/frame_%08d.png" -i "$INPUT" \
      -map 0:v:0 -map 1:a:0 \
      -c:v h264_nvenc -preset p6 -tune hq -cq 16 -b:v 0 \
      "${VF_ARGS[@]}" \
      -c:a copy -shortest \
      "$OUTPUT"
  else
    ffmpeg -hide_banner -loglevel error -y \
      -framerate "$FPS" -i "${FRAMES_UP}/frame_%08d.png" -i "$INPUT" \
      -map 0:v:0 -map 1:a:0 \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      "${VF_ARGS[@]}" \
      -c:a copy -shortest \
      "$OUTPUT"
  fi
else
  if [[ "$ENCODER" == "h264_nvenc" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -framerate "$FPS" -i "${FRAMES_UP}/frame_%08d.png" \
      -map 0:v:0 \
      -c:v h264_nvenc -preset p6 -tune hq -cq 16 -b:v 0 \
      "${VF_ARGS[@]}" \
      "$OUTPUT"
  else
    ffmpeg -hide_banner -loglevel error -y \
      -framerate "$FPS" -i "${FRAMES_UP}/frame_%08d.png" \
      -map 0:v:0 \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      "${VF_ARGS[@]}" \
      "$OUTPUT"
  fi
fi

[[ -f "$OUTPUT" ]] || { err "Output missing: $OUTPUT"; exit 6; }
log "Done: $OUTPUT"
