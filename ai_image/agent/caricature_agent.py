from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import re
from ai_image.models.vision_blip import BlipDescriber


@dataclass
class CaricatureConcept:
    title: str
    caricature_prompt: str
    caption: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "caricature_prompt": self.caricature_prompt,
            "caption": self.caption,
        }


# 1. Image → face description 
# def describe_face_stub(image: Image.Image) -> str:
#     """
#     TEMPORARY: fake face description.
#     """
#     return "a smiling person looking at the camera"
# Global BLIP describer instance (loaded once)
_blip_describer: BlipDescriber | None = None

def describe_face(image: Image.Image) -> str:
    """
    Use BLIP to generate a natural language description of the face image.
    """
    global _blip_describer
    if _blip_describer is None:
        # initialize lazily on first use
        _blip_describer = BlipDescriber(device="cpu")

    desc = _blip_describer.describe(image)
    return desc.text

# 2. User text → hobby keywords
def extract_hobbies(user_text: str) -> List[str]:
    """
    Very simple keyword extraction:
    - lowercases
    - splits on commas, 'and', and newlines
    - strips empty pieces
    """
    if not user_text:
        return []

    raw = user_text.lower()
    # Split by comma, newline, or the word "and"
    tokens = re.split(r"[,\n]|\\band\\b", raw)
    hobbies: List[str] = []

    for tok in tokens:
        tok = tok.strip()
        if len(tok) > 2:  # ignore tiny words like 'an', 'to'
            hobbies.append(tok)

    return hobbies


# 3. Face + hobbies → title, prompt, caption 
def build_caricature_concept(face_desc: str, hobbies: List[str]) -> CaricatureConcept:
    """
    Combine the face description and hobbies into:
    - a title
    - a detailed prompt for an image model
    - a short caption for the user
    """

    # Title: if hobbies exist, use the first one in a playful way
    if hobbies:
        main_hobby = hobbies[0]
        title = f"The {main_hobby.capitalize()} Caricature"
    else:
        title = "Personal Caricature"

    # Build the main prompt
    prompt = (
        f"A playful caricature of {face_desc}. "
        "Exaggerate expressive but kind facial features. "
    )

    if hobbies:
        prompt += (
            "Incorporate the user's interests: "
            + ", ".join(hobbies)
            + ". "
        )

    prompt += (
        "Use clean black line art, clear contours, minimal shading, "
        "so that the drawing can be reproduced by a robotic arm."
    )

    # Simple caption
    if hobbies:
        caption = f"A caricature inspired by your passion for {', '.join(hobbies)}."
    else:
        caption = "A playful caricature based on your photo."

    return CaricatureConcept(title=title, caricature_prompt=prompt, caption=caption)


# 4. The Agent: glue all steps together
def run_caricature_agent(image: Image.Image, user_text: str) -> Dict[str, str]:
    """
    Main function to use from outside:

    - image: PIL.Image with the user's face
    - user_text: free-form text from the user about their hobbies

    Returns:
        dict with keys: title, caricature_prompt, caption
    """
    face_desc = describe_face(image)            # (1) TEMP: stub
    hobbies = extract_hobbies(user_text)             # (2)
    concept = build_caricature_concept(face_desc, hobbies)  # (3)
    return concept.to_dict()
