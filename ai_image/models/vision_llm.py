# ai_image/models/vision_llm.py

"""
Wrapper around the vision-language model used by the caricature agent.

Right now this is a dummy implementation that does NOT call a real model.
Later, we will replace the core logic with a call to BakLLaVA / LLaVA / Qwen-VL.
"""

from dataclasses import dataclass
from typing import Optional
from PIL import Image


@dataclass
class VisionLLMResult:
    """Structured result from the vision model."""
    description: str   # A descriptive text that mixes face + interest


class VisionLLM:
    """
    High-level interface for our vision-language model.

    Usage:
        vllm = VisionLLM()
        result = vllm.describe_image_with_interest(image, "I like bouldering")
    """

    def __init__(self, model_name: str = "dummy", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # TODO: Later, load the actual model & processor/tokenizer here.
        # For now this is just a stub.

    def describe_image_with_interest(
        self,
        image: Image.Image,
        user_text: str,
        language: str = "en",
    ) -> VisionLLMResult:
        """
        Take a face image + user text and produce a descriptive text.

        For now, this is just a handcrafted dummy response so we can
        develop the rest of the pipeline.
        """
        base_desc = (
            "A friendly caricature of a person based on the provided face photo. "
            "Exaggerate expressive facial features in a kind, playful way."
        )

        if user_text:
            interest_part = f" Include elements related to: {user_text!r}."
        else:
            interest_part = " No specific hobby given; keep it generic and fun."

        # You could make this slightly dynamic so it's less boring:
        description = base_desc + interest_part

        return VisionLLMResult(description=description)
