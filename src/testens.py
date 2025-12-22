import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
#import mediapipe as mp  #GAB HIER PROBLEME

# -----------------------------
# 1. Originalbild laden
# -----------------------------
input_path = "src/assets/whiteguy2.png"
img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 2. Gesichtserkennung
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# -----------------------------
# 3. Maske erstellen
# -----------------------------
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

for (x, y, w, h) in faces:
    # Rechteck etwas größer als Gesicht
    padding = 10
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

# Maske weichzeichnen
mask = cv2.GaussianBlur(mask, (15,15), 0)

# Maske speichern
mask_pil = Image.fromarray(mask)
mask_pil.save("src/assets/face_mask.png")
print("Maskenbild gespeichert: src/assets/face_mask.png")

# -----------------------------
# 4. PIPELINE LADEN (FP16 + Low VRAM)
# -----------------------------
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.load_lora_weights("LoRAs/Urabewe_Caricature.safetensors")
#https://civitai.green/models/1506597/urabewe-caricature?
pipe.safety_checker = None
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

# -----------------------------
# 5. INPUT BILD + MASKE
# -----------------------------
init_img = Image.open("src/assets/whiteguy2.png").convert("RGB").resize((384,384))
mask_img = Image.open("src/assets/face_mask.png").convert("L").resize((384,384))

# -----------------------------
# 6. PROMPT (stärkere Karikatur)
# -----------------------------
prompt = (
    "caricature of the person in the masked area, keep facial likeness, "
    "exaggerated eyes, big nose, wide mouth, funny expression, humorous, stylized"
)

# -----------------------------
# 7. GENERATION PARAMS
# -----------------------------
result = pipe(
    prompt=prompt,
    image=init_img,
    mask_image=mask_img,
    strength=0.5,          # stärkerer Effekt
    num_inference_steps=25,
    guidance_scale=5.0
).images[0]

# -----------------------------
# 8. OUTPUT
# -----------------------------
result.save("src/assets/caricature_inpaint.png")
print("Fertig mit caricature")



##########2. Durchlauf

init_img = Image.open("src/assets/caricature_inpaint.png").convert("RGB").resize((384,384))
img_np = np.array(init_img)

# -----------------------------
# 2. Maske erstellen (nur Hände/Objektbereich = weiß)
# -----------------------------
mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

# Hier z.B. Bereich unten mittig (Hände)
h, w = mask.shape
mask[int(h*0.65):int(h*0.95), int(w*0.35):int(w*0.65)] = 255

mask = cv2.GaussianBlur(mask, (7,7), 0)
mask_pil = Image.fromarray(mask)
mask_pil.save("src/assets/hobby_mask.png")

# -----------------------------
# 3. Pipeline laden
# -----------------------------
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.load_lora_weights("LoRAs/Urabewe_Caricature.safetensors")
pipe.safety_checker = None
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

# -----------------------------
# 4. Prompt Hobby
# -----------------------------
prompt = (
    "a small acoustic guitar held naturally in the person's hands, "
    "stylized, readable, only change the hands area"
)
negative_prompt = "altered face, deformed head, extra faces, merged objects"

generator = torch.Generator("cuda").manual_seed(42)

# -----------------------------
# 5. Generation
# -----------------------------
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_img,
    mask_image=mask_pil,
    strength=0.7,  # etwas stärker, da nur Hände geändert werden
    num_inference_steps=25,
    guidance_scale=5.0,
    generator=generator
).images[0]

# -----------------------------
# 6. Speichern
# -----------------------------
result.save("src/assets/caricature_with_hobby.png")
print("Fertig! Hobby in Hände eingefügt.")






# # -------------------------------------------------------------- HOBBYMASKE GENERIEREN

# img = cv2.imread("src/assets/caricature_inpaint.png")
# h, w = img.shape[:2]

# # -----------------------------------
# # 2. Selfie-Segmentation (Personenmaske)
# # -----------------------------------
# mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
# results = mp_selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# # results.segmentation_mask = Werte 0..1
# segmask = results.segmentation_mask
# segmask = (segmask * 255).astype(np.uint8)

# # -----------------------------------
# # 3. Hobby-Maske generieren:
# #    Wir nehmen unteren Körperbereich als "weiß"
# # -----------------------------------
# mask = np.zeros((h, w), dtype=np.uint8)

# # Bereich unter dem Gesicht (ungefähr Mitte nach unten)
# # Du kannst das anpassen
# mask[int(h*0.45):int(h*0.95), :] = 255

# # Segmentierung UND Bereich kombinieren:
# mask = cv2.bitwise_and(mask, segmask)

# # Leicht weichzeichnen für natürlicheren Übergang
# mask = cv2.GaussianBlur(mask, (25,25), 0)

# # -----------------------------------
# # 4. Maske speichern
# # -----------------------------------
# mask_pil = Image.fromarray(mask)
# mask_pil.save("src/assets/hobby_mask.png")

# print("Hobby-Maske gespeichert unter src/assets/hobby_mask.png")





# # -------------------------------------------------------------- ZWEITE INPAINTING-RUN FÜR HOBBY

# # -----------------------------
# # 1. Vorheriges Ergebnis laden
# # -----------------------------
# first_result = Image.open("src/assets/caricature_inpaint.png").convert("RGB")

# # -----------------------------
# # 2. Neue Maske für Hobby erstellen
# # -----------------------------
# # Weiß = Bereich, der ersetzt wird (z.B. Hände/Platz für Objekt)
# # Schwarz = bleibt unverändert
# hobby_mask = Image.open("src/assets/hobby_mask.png").convert("L").resize(first_result.size)

# # -----------------------------
# # 3. Prompt für Hobby
# # -----------------------------
# hobby_prompt = (
#     "caricature of the person in the masked area, keep facial likeness, "
#     "exaggerated features, humorous, stylized, gaming headset on the head"
# )


# # -----------------------------
# # 4. Zweite Generierung
# # -----------------------------
# second_result = pipe(
#     prompt=hobby_prompt,
#     image=first_result,
#     mask_image=hobby_mask,
#     strength=0.7,          # Effekt nur auf Maskenbereich
#     num_inference_steps=25,
#     guidance_scale=4.5
# ).images[0]

# # -----------------------------
# # 5. Ergebnis speichern
# # -----------------------------
# second_result.save("src/assets/caricature_with_hobby.png")
# print("Fertig! Ergebnis mit Hobby gespeichert.")






# # import torch
# # from diffusers import StableDiffusionImg2ImgPipeline
# # from PIL import Image
# # import random
# # from huggingface_hub import hf_hub_download

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
