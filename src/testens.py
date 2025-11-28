import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random

# Bild laden
init_image = Image.open("src/assets/tüp.png").convert("RGB")

# Device wählen
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline laden
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "timbrooks/toonify", 
    dtype=torch.float16,
#    use_auth_token=True  # falls private Modelle
).to(device)

prompt = (
    "clean caricature portrait, stylized face, soft shading, strong outlines, "
    "exaggerated proportions, high quality"
)

# Stärke: Wie stark das Original verändert wird
strength = 0.25

# Seed: Für Zufall
seed = random.randint(0, 999999999)
generator = torch.Generator(device).manual_seed(seed)

# Inferenz
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=2.0,     # bei SDXL turbo klein halten
    generator=generator
).images[0]

# Speichern
result.save("src/assets/caricature.png")
