from google import genai
from google.genai import types
from PIL import Image

import json
from pathlib import Path 
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AGENT_JSON = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "agent_output.json"
IMAGE_PATH = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "assets" / "whiteguy2.png" #frontend bild einbinden
OUT_IMAGE = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "assets" / "caricature_result.png"

#client = genai.Client(api_key="")

def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY oder GEMINI_API_KEY fehlt")
    return genai.Client(api_key=api_key)

def main():
    print("SCRIPT PATH:", Path(__file__).resolve())
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("AGENT_JSON PATH:", AGENT_JSON)
    print("AGENT_JSON EXISTS?", AGENT_JSON.exists())

    client = get_gemini_client()

    #1. Agent Ergebnis laden 
    with open(AGENT_JSON, "r", encoding="utf-8") as f:
        agent_result = json.load(f)

    prompt = agent_result["caricature_prompt"]

    #2. Ausgangsbild laden
    input_image = Image.open(IMAGE_PATH)

    #3. Gemini Anfrage
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
        return

    # 4) Ausgabe speichern (gleicher Pfad wie SD-Ergebnis)
    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)

    for part in content.parts:
        img = part.as_image()
        if img:
            img.save(OUT_IMAGE)
            print(f" Saved to: {OUT_IMAGE}")
            break

if __name__ == "__main__":
    main()










