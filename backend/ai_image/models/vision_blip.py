# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

"""
BLIP-based image describer.

Uses the Salesforce/blip-image-captioning-base model to generate
a caption for a face photo. We then use that caption as the
face description in the caricature agent.
"""

from dataclasses import dataclass
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


@dataclass
class BlipDescription:
    text: str


class BlipDescriber:
    """
    Small wrapper around BLIP image captioning.

    Example:
        describer = BlipDescriber(device="cpu")
        desc = describer.describe(image)
        print(desc.text)
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device

        print(f"[BLIP] Loading model '{model_name}' on {device}...")

        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        ).to(device)
        self.model.eval()

    def describe(self, image: Image.Image) -> BlipDescription:
        """
        Generate a caption for the given image.
        BLIP expects a PIL RGB image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Inference
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3,
            )

        # Decode the text
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return BlipDescription(text=caption)
