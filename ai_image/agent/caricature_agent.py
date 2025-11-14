"""
Caricature Copilot Agent.

High-level job:
- Input: face image + user text (hobby/interest).
- Uses a vision-language model to get a descriptive idea.
- Wraps that into a structured caricature concept:
    - title
    - caricature_prompt (for an image generator)
    - caption (short joke for the user)
"""

from dataclasses import dataclass
from typing import Dict
from PIL import Image

from ai_image.config import VISION_LLM_MODEL_NAME, DEVICE
from ai_image.models.vision_llm import VisionLLM


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


class CaricatureAgent:
    def __init__(self):
        self.vllm = VisionLLM(model_name=VISION_LLM_MODEL_NAME, device=DEVICE)

    def run(self, image: Image.Image, user_text: str) -> CaricatureConcept:
        """
        Main entry point: given an image + hobby text,
        return a structured caricature concept.
        """

        # 1) Ask the vision model (dummy for now)
        vllm_result = self.vllm.describe_image_with_interest(
            image=image,
            user_text=user_text,
        )

        # 2) Turn the vision description into a proper "prompt" + title + caption.
        #    Right now this is simple rule-based logic.
        #    Later we can let a text-only LLM refine this if we want.
        base_prompt = vllm_result.description

        # Simple heuristic for title:
        if user_text:
            title = f"Caricature of a {user_text}"
        else:
            title = "Personal Caricature"

        # Simple caption (this can become much smarter later):
        caption = "A playful Rob Ross-style caricature based on your photo and interests."

        return CaricatureConcept(
            title=title,
            caricature_prompt=base_prompt,
            caption=caption,
        )


# Convenience function so other code can just call run_caricature_agent(...)
def run_caricature_agent(image: Image.Image, user_text: str) -> Dict[str, str]:
    agent = CaricatureAgent()
    concept = agent.run(image, user_text)
    return concept.to_dict()
