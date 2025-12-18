# sd_pipeline/run_agent_only.py
import json
from pathlib import Path
from PIL import Image
from ai_image.agent.caricature_agent import run_caricature_agent

IMAGE_PATH = "ai_image/sd_pipeline/assets/whiteguy2.png"
USER_TEXT = "I love climbing, coding and cooking"
OUT_JSON = "sd_pipeline/agent_output.json"

def main():
    image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    result = run_caricature_agent(image, USER_TEXT)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ Agent output saved to {OUT_JSON}")

if __name__ == "__main__":
    main()
