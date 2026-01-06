from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(api_key="")

# 1) kleines, harmloses Testbild laden
input_image = Image.open("test_assets/young_man.jpg")

prompt = "Create a simple black and white caricature of this person, with exaggerated eyes and head, playing football."

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt, input_image],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
    ),
)

cand = response.candidates[0]
print("finish_reason:", cand.finish_reason)

content = getattr(cand, "content", None)
if not content:
    print("No content returned, maybe blocked or failed.")
else:
    for part in content.parts:
        img = part.as_image()
        if img:
            img.save("caricature.png")
            print("saved caricature.png")
