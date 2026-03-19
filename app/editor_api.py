from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re
import uuid
import json
import shlex
import subprocess
import random
from functools import lru_cache

EDIT_ROOT = os.getenv("EDIT_ROOT", "/workspace")
EXPORT_DIR = os.path.join(EDIT_ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

DEFAULT_W = int(os.getenv("EDIT_W", "1080"))
DEFAULT_H = int(os.getenv("EDIT_H", "1920"))
DEFAULT_FPS = float(os.getenv("EDIT_FPS", "30"))
DEFAULT_TRANS = float(os.getenv("EDIT_TRANS_DUR", "0.12"))
DEFAULT_CRF = int(os.getenv("EDIT_CRF", "18"))
DEFAULT_PRESET = os.getenv("EDIT_PRESET", "veryfast")
DEFAULT_SUB_FONT = os.getenv("EDIT_SUB_FONT", "DejaVu Sans")


class Clip(BaseModel):
    path: str
    audio: Optional[bool] = True
    audio_volume: Optional[float] = 1.0


class SubtitleRequest(BaseModel):
    text: str
    style: Optional[str] = "auto"
    mode: Optional[str] = "chunk"
    words_per_caption: Optional[int] = 3
    uppercase: Optional[bool] = False


class EditRequest(BaseModel):
    clips: List[Clip]
    output_name: Optional[str] = None
    transition: Optional[float] = None
    transition_style: Optional[str] = "cut"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    crf: Optional[int] = None
    preset: Optional[str] = None
    audio_path: Optional[str] = None
    audio_start: Optional[float] = None
    audio_volume: Optional[float] = None
    audio_trim_to_video: Optional[bool] = True
    subtitles: Optional[SubtitleRequest] = None


def _run(cmd: List[str]) -> str:
    # Debug: zeigt dir das exakte ffmpeg Kommando in Logs
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout


def _probe_duration(path: str) -> float:
    # Stream-duration ist oft stabiler als format-duration bei KI-Clips
    for args in (
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path],
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
    ):
        try:
            out = subprocess.check_output(args, text=True).strip()
            d = float(out)
            if d > 0.001:
                return d
        except Exception:
            pass
    return 0.0


def _probe_format_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
            text=True,
        ).strip()
        d = float(out)
        return d if d > 0.001 else 0.0
    except Exception:
        return 0.0


def _has_audio(path: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "json", path],
            text=True,
        )
        data = json.loads(out)
        return bool(data.get("streams"))
    except Exception:
        return False


def _sanitize_output_name(name: Optional[str], job_id: str) -> str:
    n = (name or "").strip()
    if n.lower() in ("", "undefined", "[undefined]", "null", "[null]", "none", "[none]"):
        n = f"edit_{job_id}"
    n = re.sub(r"[^\w\-. ]+", "_", n).strip()
    if not n.lower().endswith(".mp4"):
        n += ".mp4"
    return n


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _ass_color(hex_rgb: str) -> str:
    rgb = hex_rgb.strip().lstrip("#")
    if len(rgb) != 6:
        rgb = "FFFFFF"
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&"


SUBTITLE_STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "clean_premium": {
        "fontname": DEFAULT_SUB_FONT,
        "fontsize": 72,
        "primary": _ass_color("FFFFFF"),
        "secondary": _ass_color("A7F3D0"),
        "outline": _ass_color("101010"),
        "back": _ass_color("101010"),
        "bold": -1,
        "outline_w": 5,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 210,
        "anim": r"{\fad(70,90)\fscx112\fscy112\t(0,140,\fscx100\fscy100)}",
    },
    "tiktok_punch": {
        "fontname": DEFAULT_SUB_FONT,
        "fontsize": 82,
        "primary": _ass_color("FFF200"),
        "secondary": _ass_color("FF5C8A"),
        "outline": _ass_color("111111"),
        "back": _ass_color("111111"),
        "bold": -1,
        "outline_w": 7,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 220,
        "anim": r"{\fad(40,70)\blur0.3\fscx126\fscy126\t(0,120,\fscx100\fscy100)}",
    },
    "neon_hook": {
        "fontname": DEFAULT_SUB_FONT,
        "fontsize": 78,
        "primary": _ass_color("F8FAFC"),
        "secondary": _ass_color("22D3EE"),
        "outline": _ass_color("0F172A"),
        "back": _ass_color("0F172A"),
        "bold": -1,
        "outline_w": 6,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 205,
        "anim": r"{\fad(50,70)\fscx118\fscy118\t(0,110,\fscx100\fscy100)}",
    },
    "hook_alert": {
        "fontname": DEFAULT_SUB_FONT,
        "fontsize": 84,
        "primary": _ass_color("FFFFFF"),
        "secondary": _ass_color("FF4D4D"),
        "outline": _ass_color("111111"),
        "back": _ass_color("111111"),
        "bold": -1,
        "outline_w": 8,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 230,
        "anim": r"{\fad(30,60)\fscx132\fscy132\t(0,110,\fscx100\fscy100)}",
    },
}

XFADER_TRANSITIONS: Dict[str, str] = {
    "fade": "fade",
    "dissolve": "dissolve",
    "smoothleft": "smoothleft",
    "smoothright": "smoothright",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "circleopen": "circleopen",
    "zoomin": "zoomin",
}


def _pick_subtitle_style(style_name: Optional[str], seed_value: str) -> tuple[str, Dict[str, Any]]:
    style_key = (style_name or "auto").strip().lower()
    if style_key in ("", "auto", "random"):
        keys = sorted(SUBTITLE_STYLE_PRESETS.keys())
        rng = random.Random(seed_value)
        style_key = rng.choice(keys)
    if style_key not in SUBTITLE_STYLE_PRESETS:
        style_key = "clean_premium"
    return style_key, SUBTITLE_STYLE_PRESETS[style_key]


def _resolve_transition_style(style_name: Optional[str]) -> tuple[str, Optional[str]]:
    style_key = (style_name or "cut").strip().lower()
    if style_key in ("", "none"):
        style_key = "cut"
    if style_key in ("cut", "dip_black", "dip_white"):
        return style_key, None
    return style_key if style_key in XFADER_TRANSITIONS else "cut", XFADER_TRANSITIONS.get(style_key)


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _split_caption_chunks(text: str, mode: str, words_per_caption: int) -> List[str]:
    words = [w for w in text.strip().split() if w]
    if not words:
        return []
    if mode == "word":
        return words

    target = max(1, int(words_per_caption))
    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(current) >= target:
            chunks.append(" ".join(current))
            current = []
            continue
        if re.search(r"[.!?:,;…]$", word) and len(current) >= max(1, target - 1):
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


@lru_cache(maxsize=1)
def _get_asr_pipeline():
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cpu",
    )


def _normalize_caption_words(text: str) -> List[str]:
    return [w for w in text.strip().split() if w]


def _clean_asr_segments(raw_chunks: List[Dict[str, Any]], audio_duration: float) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    last_end = 0.0
    for chunk in raw_chunks:
        text = str(chunk.get("text") or "").strip()
        ts = chunk.get("timestamp") or ()
        start = float(ts[0]) if len(ts) > 0 and ts[0] not in (None, "") else last_end
        start = max(0.0, start, last_end)
        end = ts[1] if len(ts) > 1 else None
        end_f = float(end) if end not in (None, "") else None
        if not text:
            continue
        if end_f is None or end_f <= start:
            end_f = None
        segments.append({"start": max(0.0, start), "end": end_f, "text": text})
        if end_f is not None:
            last_end = end_f

    for i, seg in enumerate(segments):
        if seg["end"] is None:
            if i + 1 < len(segments):
                seg["end"] = max(seg["start"], segments[i + 1]["start"])
            else:
                seg["end"] = max(seg["start"], audio_duration)

    cleaned: List[Dict[str, Any]] = []
    for seg in segments:
        if seg["end"] <= seg["start"]:
            continue
        cleaned.append(seg)

    if cleaned and cleaned[-1]["end"] < audio_duration:
        cleaned[-1]["end"] = audio_duration
    return cleaned


def _asr_segmented_chunks(
    subtitle_req: SubtitleRequest,
    audio_path: str,
    audio_start: float = 0.0,
    max_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    audio_duration = _probe_format_duration(audio_path)
    if audio_duration <= 0.0:
        return []

    try:
        pipe = _get_asr_pipeline()
        result = pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs={"language": "german", "task": "transcribe"},
        )
    except Exception:
        return []

    segments = _clean_asr_segments(result.get("chunks", []), audio_duration)
    if not segments:
        return []

    window_start = max(0.0, float(audio_start))
    window_end = audio_duration if max_duration is None else min(audio_duration, window_start + max(0.0, float(max_duration)))
    full_window = window_start <= 0.001 and abs(window_end - audio_duration) <= 0.05
    clipped_segments: List[Dict[str, Any]] = []
    for seg in segments:
        if seg["end"] <= window_start or seg["start"] >= window_end:
            continue
        clipped_segments.append({
            "start": max(seg["start"], window_start) - window_start,
            "end": min(seg["end"], window_end) - window_start,
            "text": seg["text"],
        })
    segments = [seg for seg in clipped_segments if seg["end"] > seg["start"]]
    if not segments:
        return []

    original_words = _normalize_caption_words(subtitle_req.text)
    if not original_words:
        return []

    mode = (subtitle_req.mode or "chunk").strip().lower()
    words_per_caption = max(1, int(subtitle_req.words_per_caption or 3))
    out: List[Dict[str, Any]] = []
    word_cursor = 0

    for idx, seg in enumerate(segments):
        remaining_words = len(original_words) - word_cursor
        if remaining_words <= 0:
            break

        asr_word_count = max(1, len(_normalize_caption_words(seg["text"])))
        take = remaining_words if full_window and idx == len(segments) - 1 else min(remaining_words, asr_word_count)
        seg_words = original_words[word_cursor:word_cursor + take]
        word_cursor += take
        if not seg_words:
            continue

        if mode == "word":
            seg_chunks = seg_words
        else:
            seg_chunks = [" ".join(seg_words[i:i + words_per_caption]) for i in range(0, len(seg_words), words_per_caption)]

        duration = max(0.06, seg["end"] - seg["start"])
        counts = [max(1, len(c.split())) for c in seg_chunks]
        total = sum(counts)
        cursor = seg["start"]
        for chunk_idx, chunk in enumerate(seg_chunks):
            portion = duration * (counts[chunk_idx] / total)
            end = seg["end"] if chunk_idx == len(seg_chunks) - 1 else min(seg["end"], cursor + portion)
            out.append({"start": cursor, "end": end, "text": chunk})
            cursor = end

    if full_window and word_cursor < len(original_words) and out:
        tail = " ".join(original_words[word_cursor:])
        out[-1]["text"] = f"{out[-1]['text']} {tail}".strip()
        out[-1]["end"] = max(out[-1]["end"], audio_duration)

    return [chunk for chunk in out if chunk["end"] > chunk["start"]]


def _highlight_last_word(chunk: str, primary_color: str, secondary_color: str) -> str:
    parts = chunk.split()
    if len(parts) <= 1:
        return rf"{{\c{secondary_color}}}{_ass_escape(chunk)}{{\c{primary_color}}}"
    lead = " ".join(parts[:-1])
    tail = parts[-1]
    return f"{_ass_escape(lead)} " + rf"{{\c{secondary_color}}}{_ass_escape(tail)}{{\c{primary_color}}}"


def _build_ass_subtitles(
    subtitle_req: SubtitleRequest,
    ass_path: str,
    width: int,
    height: int,
    total_duration: float,
    seed_value: str,
    audio_path: Optional[str] = None,
    audio_start: float = 0.0,
    audio_max_duration: Optional[float] = None,
) -> str:
    style_name, style = _pick_subtitle_style(subtitle_req.style, seed_value)
    mode = (subtitle_req.mode or "chunk").strip().lower()
    timed_chunks = (
        _asr_segmented_chunks(subtitle_req, audio_path, audio_start=audio_start, max_duration=audio_max_duration)
        if audio_path
        else []
    )
    if not timed_chunks:
        chunks = _split_caption_chunks(subtitle_req.text, mode, subtitle_req.words_per_caption or 3)
        if not chunks:
            return style_name
        chunk_word_counts = [max(1, len(c.split())) for c in chunks]
        total_words = sum(chunk_word_counts)
        timed_chunks = []
        cursor = 0.0
        for idx, chunk in enumerate(chunks):
            proportional = total_duration * (chunk_word_counts[idx] / total_words)
            end = total_duration if idx == len(chunks) - 1 else min(total_duration, cursor + proportional)
            timed_chunks.append({"start": cursor, "end": end, "text": chunk})
            cursor = end

    if not timed_chunks:
        return style_name
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Main,{fontname},{fontsize},{primary},{secondary},{outline},{back},{bold},0,0,0,100,100,0,0,1,{outline_w},{shadow},{alignment},50,50,{margin_v},1".format(
            **style
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    for chunk in timed_chunks:
        display = chunk["text"].upper() if subtitle_req.uppercase else chunk["text"]
        display = _highlight_last_word(display, style["primary"], style["secondary"])
        lines.append(
            "Dialogue: 0,{start},{end},Main,,0,0,0,,{anim}{text}".format(
                start=_format_ass_time(chunk["start"]),
                end=_format_ass_time(chunk["end"]),
                anim=style["anim"],
                text=display,
            )
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return style_name


def render_edit(req: EditRequest) -> Dict[str, Any]:
    if not req.clips:
        return {"ok": False, "error": "no_clips"}

    # Settings
    W = req.width or DEFAULT_W
    H = req.height or DEFAULT_H
    FPS = float(req.fps or DEFAULT_FPS)
    TRANS = float(req.transition if req.transition is not None else DEFAULT_TRANS)
    TRANS = max(0.0, min(TRANS, 1.0))
    TRANS_STYLE, XFADE_NAME = _resolve_transition_style(req.transition_style)
    CRF = int(req.crf or DEFAULT_CRF)
    PRESET = req.preset or DEFAULT_PRESET
    AUDIO_START = max(0.0, float(req.audio_start or 0.0))
    AUDIO_VOL = _clamp(float(req.audio_volume if req.audio_volume is not None else 1.0), 0.0, 1.0)
    AUDIO_TRIM = True if req.audio_trim_to_video is None else bool(req.audio_trim_to_video)

    job_id = uuid.uuid4().hex[:8]
    out_name = _sanitize_output_name(req.output_name, job_id)
    out_path = os.path.join(EXPORT_DIR, out_name)
    tmp_out_path = os.path.join(EXPORT_DIR, f".tmp_{job_id}_{out_name}")
    sub_ass_path = os.path.join(EXPORT_DIR, f".subs_{job_id}.ass")

    paths = []
    durs = []
    auds = []
    clip_audio_vols = []

    for c in req.clips:
        p = c.path
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        paths.append(p)
        d = _probe_duration(p)
        if d <= 0.001:
            d = 5.0
        durs.append(d)
        auds.append(_has_audio(p) and (True if c.audio is None else bool(c.audio)))
        clip_audio_vols.append(_clamp(float(c.audio_volume if c.audio_volume is not None else 1.0), 0.0, 1.0))

    audio_path = None
    if req.audio_path not in (None, ""):
        audio_path = req.audio_path
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)

    # ffmpeg inputs
    cmd = ["ffmpeg", "-y"]
    for p in paths:
        cmd += ["-i", p]
    if audio_path:
        cmd += ["-i", audio_path]

    # filter_complex: normalize -> optional smart transitions -> optional audio overlay
    fc_parts = []

    for i in range(len(paths)):
        # Video normalize (CFR, gleiche Größe, gleiche SAR, stabile PTS)
        fc_parts.append(
            f"[{i}:v]"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={FPS},"
            f"format=yuv420p,"
            f"setpts=PTS-STARTPTS"
            f"[v{i}]"
        )

        # Audio normalize oder Silence-Fallback passend zur Clip-Dauer
        if auds[i]:
            audio_chain = (
                f"[{i}:a]"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"aresample=async=1:first_pts=0,"
                f"asetpts=PTS-STARTPTS"
            )
            if clip_audio_vols[i] != 1.0:
                audio_chain += f",volume={clip_audio_vols[i]:.3f}"
            fc_parts.append(
                audio_chain + f"[a{i}]"
            )
        else:
            fc_parts.append(
                f"anullsrc=r=48000:cl=stereo:d={durs[i]}[a{i}]"
            )

    video_out_label = "[v0]"
    audio_out_label = "[a0]"
    total_dur = durs[0] if durs else 0.0

    if len(paths) > 1:
        if XFADE_NAME and TRANS > 0:
            video_out_label = "[v0]"
            audio_out_label = "[a0]"
            total_dur = durs[0]
            for i in range(1, len(paths)):
                next_v = f"[v{i}]"
                next_a = f"[a{i}]"
                vout = f"[vx{i}]"
                aout = f"[ax{i}]"
                offset = max(0.0, total_dur - TRANS)
                fc_parts.append(
                    f"{video_out_label}{next_v}"
                    f"xfade=transition={XFADE_NAME}:duration={TRANS}:offset={offset:.6f}"
                    f"{vout}"
                )
                fc_parts.append(
                    f"{audio_out_label}{next_a}"
                    f"acrossfade=d={TRANS}:c1=tri:c2=tri"
                    f"{aout}"
                )
                video_out_label = vout
                audio_out_label = aout
                total_dur += durs[i] - TRANS
        else:
            concat_inputs = []
            for i in range(len(paths)):
                concat_inputs += [f"[v{i}]", f"[a{i}]"]
                if i != len(paths) - 1 and TRANS > 0 and TRANS_STYLE != "cut":
                    dip_color = "white" if TRANS_STYLE == "dip_white" else "black"
                    fc_parts.append(f"color=c={dip_color}:s={W}x{H}:r={FPS}:d={TRANS}[vb{i}]")
                    fc_parts.append(f"anullsrc=r=48000:cl=stereo:d={TRANS}[ab{i}]")
                    concat_inputs += [f"[vb{i}]", f"[ab{i}]"]

            n_segments = len(concat_inputs) // 2  # (v,a) Paare
            fc_parts.append(
                "".join(concat_inputs) + f"concat=n={n_segments}:v=1:a=1[vout][aout]"
            )
            video_out_label = "[vout]"
            audio_out_label = "[aout]"
            total_dur = sum(durs) + (TRANS * max(0, len(paths) - 1) if TRANS_STYLE != "cut" else 0.0)
    else:
        video_out_label = "[v0]"
        audio_out_label = "[a0]"
        total_dur = durs[0] if durs else 0.0

    final_audio_label = audio_out_label
    if audio_path:
        audio_input_idx = len(paths)
        overlay_chain = (
            f"[{audio_input_idx}:a]"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"aresample=async=1:first_pts=0,"
        )
        if AUDIO_TRIM:
            overlay_chain += f"atrim=start={AUDIO_START}:end={AUDIO_START + total_dur},"
        elif AUDIO_START > 0:
            overlay_chain += f"atrim=start={AUDIO_START},"
        overlay_chain += "asetpts=PTS-STARTPTS"
        if AUDIO_VOL != 1.0:
            overlay_chain += f",volume={AUDIO_VOL:.3f}"
        fc_parts.append(overlay_chain + "[aext]")
        fc_parts.append(f"{audio_out_label}[aext]amix=inputs=2:duration=first:dropout_transition=0[aoutmix]")
        final_audio_label = "[aoutmix]"

    filter_complex = ";".join(fc_parts)

    # Re-encode (verhindert Macroblock-Müll an Cuts)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", video_out_label,
        "-map", final_audio_label,
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-r", str(int(round(FPS))),
        "-g", str(int(round(FPS))),      # 1s GOP -> sehr saubere Cuts
        "-keyint_min", "1",
        "-sc_threshold", "40",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        tmp_out_path if req.subtitles and req.subtitles.text.strip() else out_path,
    ]

    _run(cmd)

    subtitle_style_used = None
    if req.subtitles and req.subtitles.text.strip():
        subtitle_style_used = _build_ass_subtitles(
            req.subtitles,
            sub_ass_path,
            W,
            H,
            total_dur,
            seed_value=job_id,
            audio_path=audio_path,
            audio_start=AUDIO_START,
            audio_max_duration=total_dur if AUDIO_TRIM else None,
        )
        _run([
            "ffmpeg",
            "-y",
            "-i",
            tmp_out_path,
            "-vf",
            f"ass={sub_ass_path}",
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            out_path,
        ])
        try:
            os.remove(tmp_out_path)
        except FileNotFoundError:
            pass

    rel = os.path.relpath(out_path, EDIT_ROOT)
    return {
        "ok": True,
        "output_path": rel,
        "output_name": out_name,
        "transition": TRANS,
        "audio_overlay": bool(audio_path),
        "transition_style": TRANS_STYLE,
        "subtitle_style": subtitle_style_used,
    }
