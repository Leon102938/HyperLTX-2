#!/usr/bin/env python3
import os
import torch
from diffusers import ZImagePipeline

def find_model_dir() -> str:
    env = os.getenv("ZIMAGE_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    candidates = [
        "/workspace/models/zimage",
        "/workspace/models/Tongyi-MAI/Z-Image",
        "/workspace/Tongyi-MAI/Z-Image",
        "/models/Tongyi-MAI/Z-Image",
        "/models/zimage",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "Tongyi-MAI/Z-Image"

def round_to_multiple(x: int, m: int = 16) -> int:
    return ((x + m - 1) // m) * m

def main() -> None:
    model = find_model_dir()
    out = os.getenv("ZIMAGE_OUT", "glass_apple2_9x16.png")

    width = round_to_multiple(int(os.getenv("ZIMAGE_W", "1536")), 16)
    height = round_to_multiple(int(os.getenv("ZIMAGE_H", "2752")), 16)

    steps = int(os.getenv("ZIMAGE_STEPS", "35"))
    guidance = float(os.getenv("ZIMAGE_GUIDANCE", "6.0"))
    seed = int(os.getenv("ZIMAGE_SEED", "50"))
    cfg_norm = os.getenv("ZIMAGE_CFG_NORM", "false").lower() == "true"

    prompt = os.getenv("ZIMAGE_PROMPT", "Ultra-realistic professional product photography in 9:16 portrait format. A high-end chef's knife, held by a hand in a professional black nitrile glove, is gently placed against the top of a perfectly smooth, solid glass apple. The blade is just touching the surface, poised but not cutting. The apple is crystal clear and glossy, with realistic swirls of deep ruby red and lime green embedded deep inside the glass, mimicking apple skin with clean color separation. The apple rests on a dark walnut wooden cutting board with a rich, detailed grain. Background is a moody, out-of-focus rustic kitchen with warm wooden tones and a shallow depth of field. Cinematic studio lighting with sharp highlights on the knife's edge and brilliant light refraction through the glass. Hyper-detailed, 8K resolution, luxury ASMR aesthetic, calm and elegant mood.")

    negative = os.getenv("ZIMAGE_NEG","ugly, deformed hands, extra fingers, missing fingers, malformed knife, organic fruit texture, fruit pulp, juice, liquid, bubbles in glass, scratches, cracks, blurry subject, watermark, text, signature, low quality, grainy, messy background, bright neon colors, unrealistic reflections, knife cutting through, broken glass, plastic look, cartoonish, 3d render style, low resolution, out of focus apple.")

    print(f"[Z-Image] Using model: {model}")
    print(f"[Z-Image] Settings: {width}x{height} steps={steps} guidance={guidance} seed={seed} cfg_norm={cfg_norm} out={out}")

    pipe = ZImagePipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to("cuda")

    gen = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        cfg_normalization=cfg_norm,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    ).images[0]

    image.save(out)
    print(f"[Z-Image] Saved: {out}")

if __name__ == "__main__":
    main()
