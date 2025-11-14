# ai_image/scripts/demo_caricature_cli.py

"""
Small CLI to test the CaricatureAgent outside the main app.

Usage:
    python -m ai_image.scripts.demo_caricature_cli
"""

from pathlib import Path
from PIL import Image

from ai_image.agent.caricature_agent import run_caricature_agent


def main():
    img_path_str = input("Path to face photo image file: ").strip()
    user_text = input("Describe your hobby / interest: ").strip()

    img_path = Path(img_path_str)
    if not img_path.is_file():
        print(f"File not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")

    result = run_caricature_agent(image=image, user_text=user_text)

    print("\n=== Caricature Agent Result ===")
    print("Title:", result["title"])
    print("Caption:", result["caption"])
    print("\nCaricature prompt:\n")
    print(result["caricature_prompt"])


if __name__ == "__main__":
    main()
