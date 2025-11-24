# ai_image/agent/caricature_agent.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from PIL import Image
import re

from ai_image.models.vision_blip import BlipDescriber


# ------------------------------
# Data structure for agent output
# ------------------------------

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


# ------------------------------
# BLIP loader + fallback
# ------------------------------

_blip_instance: Optional[BlipDescriber] = None


def _get_blip() -> BlipDescriber:
    """
    Lazy load a single BLIP instance so the model is loaded once.
    """
    global _blip_instance
    if _blip_instance is None:
        _blip_instance = BlipDescriber(device="cpu")
    return _blip_instance


def describe_face(image: Image.Image) -> str:
    """
    Describe the face using BLIP.

    If BLIP fails for any reason (model missing, corrupted download, etc.),
    fall back to a simple generic description so the agent never crashes.
    """
    try:
        blip = _get_blip()
        desc = blip.describe(image)

        out = (desc.text or "").strip()
        if not out:
            return "a person looking at the camera"

        return out

    except Exception as e:
        print("⚠️  Warning: BLIP describe_face failed — using fallback:", e)
        return "a person looking at the camera"


# ------------------------------
# English hobby extraction
# ------------------------------

ENGLISH_STOPWORDS = {
    "i", "my", "me", "and", "or", "also", "very", "really", "quite",
    "love", "like", "enjoy", "am", "is", "are", "to", "a", "the"
}


def extract_hobbies(user_text: str) -> List[str]:
    """
    Extracts hobby-related phrases from English text.

    Examples:
      "I like bouldering and baking cakes"
          → ["bouldering", "baking cakes"]

      "climbing, drawing, cooking"
          → ["climbing", "drawing", "cooking"]
    """

    if not user_text:
        return []

    text = user_text.lower().strip()

    # split by commas, semicolons, line breaks, and 'and'
    chunks = re.split(r"[,;\n]+|\band\b", text)
    hobbies = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # remove filler words
        words = [w for w in re.split(r"\s+", chunk) if w not in ENGLISH_STOPWORDS]
        phrase = " ".join(words).strip()

        if len(phrase) > 2:
            hobbies.append(phrase)

    # remove duplicates, keep order
    seen = set()
    unique = []
    for h in hobbies:
        if h not in seen:
            seen.add(h)
            unique.append(h)

    return unique


# ------------------------------
# Build the caricature concept
# ------------------------------

def build_caricature_concept(face_desc: str, hobbies: List[str]) -> CaricatureConcept:
    """
    Combines:
      - BLIP face description
      - extracted hobby phrases
    into:
      - title
      - caption
      - a Stable Diffusion-style prompt
    """

    # Title
    if hobbies:
        main = hobbies[0].title()
        title = f"The {main} Caricature"
    else:
        title = "Personal Caricature"

    # Prompt Composition
    prompt_parts = [
        f"A creative caricature of {face_desc}.",
        "Emphasize expressive but friendly facial features.",
    ]

    if hobbies:
        prompt_parts.append(
            "Incorporate these interests: " + ", ".join(hobbies) + "."
        )

    # style — optimized for robotic arm drawing
    prompt_parts.append(
        "Style: clean black line art, clear outlines, minimal shading; "
        "easy for a robotic arm to draw as a single-line or contour sketch."
    )

    prompt = " ".join(prompt_parts)

    # Caption
    if hobbies:
        caption = f"A caricature inspired by your interest in {', '.join(hobbies)}."
    else:
        caption = "A playful caricature based on your photo."

    return CaricatureConcept(
        title=title,
        caricature_prompt=prompt,
        caption=caption,
    )


# ------------------------------
# PUBLIC ENTRYPOINT
# ------------------------------

def run_caricature_agent(image: Image.Image, user_text: str) -> Dict[str, str]:
    """
    Main entrypoint for your backend.

    Input:
      - image (PIL.Image)
      - user_text (string)

    Output:
      dict with:
        - title
        - caption
        - caricature_prompt
    """

    face_desc = describe_face(image)
    hobbies = extract_hobbies(user_text)
    concept = build_caricature_concept(face_desc, hobbies)

    return concept.to_dict()
