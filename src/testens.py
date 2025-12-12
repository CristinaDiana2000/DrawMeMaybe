import requests

api_url = "https://api.deepai.org/api/cartoon-gan"
api_key = "DEIN_API_KEY"

with open("src/assets/whiteguy2.png", "rb") as f:
    response = requests.post(
        api_url,
        files={"image": f},
        headers={"api-key": api_key}
    )

data = response.json()
print(data)  # Zeigt die komplette Antwort, um die richtigen Keys zu sehen

# Beispiel, wenn die API 'output' als Liste zurückgibt
if "output" in data and len(data["output"]) > 0:
    image_url = data["output"][0]
else:
    raise ValueError("Kein Bild in der Antwort gefunden")



# import torch
# from diffusers import StableDiffusionImg2ImgPipeline
# from PIL import Image
# import random
# from huggingface_hub import hf_hub_download

# # LoRA-Datei herunterladen
# #path = hf_hub_download(
# #    repo_id="Margaritahse1/cartoon_style_LoRA", 
# #   filename="cartoon_style_LoRA.safetensors"
# #)

# #print("LoRA gespeichert unter:", path)

# print(torch.__version__)
# print(torch.cuda.is_available())

# # Bild laden
# init_image = Image.open("src/assets/whiteguy2.png").convert("RGB").resize((512,512))

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Pipeline laden
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# ).to("cuda")
# pipe.safety_checker = None #sonst heult er wegen nsfw rum

# # Besserer Karikatur-Prompt 
# prompt = (
#     "A caricature portrait, exaggerated features, fun style, cartoonish face, big eyes, wide smile"
# )

# # Werte tun gut für Karikaturen
# strength = 0.35          
# guidance_scale = 5.5     
# seed = random.randint(0, 1_000_000_000)
# generator = torch.Generator(device).manual_seed(seed)

# # Bild generieren
# result = pipe(
#     prompt=prompt,
#     image=init_image,
#     strength=strength,
#     guidance_scale=guidance_scale,
#     generator=generator
# ).images[0]

# # Speichern
# result.save("src/assets/caricature_result.png")
# print("Fertig! Seed:", seed)
