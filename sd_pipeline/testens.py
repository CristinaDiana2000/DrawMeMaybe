import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random

# Inputbild 
init_image = Image.open("src/assets/whiteguy2.png").convert("RGB")
init_image = init_image.resize((512, 512))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline laden 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  
).to(device)

# Prompt definieren 
prompt = (
    #"funny caricature, huge head, tiny body, exaggerated features, cartoon style, bold outlines, simple shapes"
    

)

# img2img Parameter
strength = 0.7           # wie stark das Original verändert wird
guidance_scale = 7.5      # wie stark das Modell dem Prompt folgt
seed = random.randint(0, 999999999)
generator = torch.Generator(device).manual_seed(seed)

# Inferenz
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# Speichern und anzeigen
result.save("src/assets/caricature_result.png")





