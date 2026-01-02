import json
import random
from pathlib import Path
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

#AGENT_JSON = "backend/ai_image/sd_pipeline/agent_output.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


AGENT_JSON = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "agent_output.json"
IMAGE_PATH = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "assets" / "whiteguy2.png"
OUT_IMAGE = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "assets" / "caricature_result.png"

print("SCRIPT PATH:", Path(__file__).resolve())
print("PROJECT_ROOT:", PROJECT_ROOT)
print("AGENT_JSON PATH:", AGENT_JSON)
print("AGENT_JSON EXISTS?", AGENT_JSON.exists())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for SD: {device}")

    # Agent-Ergebnis laden
    with open(AGENT_JSON, "r", encoding="utf-8") as f:
        agent_result = json.load(f)

    prompt = agent_result["caricature_prompt"]

    init_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    strength = 0.7
    guidance_scale = 7.5
    seed = random.randint(0, 999_999_999)
    generator = torch.Generator(device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=25,
        ).images[0]

    Path(OUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
    result.save(OUT_IMAGE)
    print(f"✅ Saved to: {OUT_IMAGE}")

if __name__ == "__main__":
    main()
