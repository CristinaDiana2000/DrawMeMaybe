from pathlib import Path
from io import BytesIO
import os

from openai import OpenAI
from PIL import Image
import requests

# Basisordner = Ordner dieser Datei
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "test_assets"
ASSETS_DIR.mkdir(exist_ok=True)

input_path = ASSETS_DIR / "red_head.png"
output_png_path = ASSETS_DIR / "red_head_out.png"
edited_path = ASSETS_DIR / "edited_image.png"

# Bild nach PNG konvertieren
img = Image.open(input_path).convert("RGBA")
img.save(output_png_path, format="PNG")

# OpenAI-Client (Key aus Umgebung)
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

client = OpenAI(api_key="")

prompt = (
    "black and white line-art caricature of the person, "
    "very exaggerated head and facial features, small body, "
    "clearly visible small guitar in their hands, "
    "simple white background, high contrast, no photo realism"
)

with open(output_png_path, "rb") as f:
    response = client.images.edit(
        model="gpt-image-1",
        image=f,
        prompt=prompt,
        size="1024x1024",
    )

if response.data and len(response.data) > 0:
    image_url = response.data[0].url
    if image_url:
        res = requests.get(image_url)
        img = Image.open(BytesIO(res.content))
        img.save(edited_path)
    else:
        print("Error: No URL in image data")
else:
    print("Error: No image data in response")

print(f"Fertig! Bild gespeichert unter: {edited_path}")
