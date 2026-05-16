import os
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

model_id = os.environ.get("HIDREAM_MODEL_ID", "HiDream-ai/HiDream-O1-Dev")
out_path = Path("/workspace/manual_hidream_test/hidream_glass_apple_01.png")

prompt = (
    "ultra-detailed photorealistic product shot of a transparent glass apple, "
    "a sharp kitchen knife approaching from the left and gently pressed against the left side of the glass apple, "
    "the blade is only positioned to cut, not actually slicing, "
    "no cut, no crack, no fracture, no broken glass, "
    "beautiful reflections and refractions inside the glass apple, "
    "clean premium studio setup, subtle dramatic lighting, soft shadows, "
    "sharp focus on the apple and knife edge, shallow depth of field, "
    "luxury commercial photography, highly detailed, crisp edges, clean background, "
    "no text, no logo, no watermark"
)

width = 1024
height = 1024
steps = 28
guidance_scale = 0.0
seed = 12345

print("MODEL:", model_id)
print("CUDA:", torch.cuda.is_available())
print("OUT:", out_path)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

img = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=gen,
).images[0]

out_path.parent.mkdir(parents=True, exist_ok=True)
img.save(out_path)

print("SAVED:", out_path)
print("SIZE:", out_path.stat().st_size)
print("SECONDS:", round(time.time() - t0, 2))
