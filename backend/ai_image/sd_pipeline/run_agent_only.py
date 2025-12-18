# sd_pipeline/run_agent_only.py

import json
from pathlib import Path
from PIL import Image
from ai_image.agent.caricature_agent import run_caricature_agent


# Resolve project root: the directory that contains "backend" and "ai_image"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths for input image and output JSON (anchored at project root)
IMAGE_PATH = PROJECT_ROOT /"ai_image" / "sd_pipeline" / "assets" / "whiteguy2.png"
OUT_JSON   = PROJECT_ROOT / "ai_image" / "sd_pipeline" / "agent_output.json"

# Temporary hardcoded user text (later replace with frontend input)
USER_TEXT = "I love climbing, coding and cooking"


def main():
    # Load and preprocess the init image
    image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))

    # Run the caricature agent (Ollama side)
    result = run_caricature_agent(image, USER_TEXT)

    # Ensure output directory exists and save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ Agent output saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
