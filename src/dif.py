import torch
from diffusers import StableDiffusionImg2ImgPipeline as sdi2ip
import transformers
from PIL import Image
import cv2 as cv
import random


init_image = Image.open("src/assets/tüp.png").convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = "Norod78/sd15-caricature-portraits-blip-captions"

pipe = sdi2ip.from_pretrained(model, torch_dtype=torch.float16).to(device)

prompt = "clean caricature portrait, sylized face, soft shading, strong outline, expressive exaggerated proportions"

strength = random.uniform(0.45 , 0.7)    #strength, how much the image does not resemble the original
seed = random.randint(0, 999999999)     #jedest random seed for variation starting point
generator = torch.Generator(device).manual_seed(seed)

result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=7.5,
                generator=generator
             ).images[0]

result.save("src/assets/anything-v5_ergebnis.png")





