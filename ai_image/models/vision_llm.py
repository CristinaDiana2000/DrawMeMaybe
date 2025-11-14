from dataclasses import dataclass #for VisionLLMResult
from typing import Optional
from PIL import Image #for image input
import torch #for tensors and sending things to CPU/GPU
from transformers import AutoProcessor #handles image+text preprocessing
from transformers import AutoModelForCausalLM #generic class for models that generate text


@dataclass          # container obejct for the model's output
class VisionLLMResult:
    description: str


class VisionLLM:
    """
    Wrapper for a small vision-language model.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base-ft", #huggingFace model name
        device: str = "cpu", # "cpu" or "cuda" depending on the hardware
    ):
    
        self.device = device
        self.model_name = model_name

        print(f"[VisionLLM] Loading model '{model_name}' on {device}...")

        # NOTE: Florence-2 is a vision-text model that can be used as a VLM.
        # If we change the model, we may need to adjust how prompts are formed.

        self.processor = AutoProcessor.from_pretrained(model_name) #downloads pprocessor configs e.g image preprocessor
        self.model = AutoModelForCausalLM.from_pretrained(          #downloads model weights and architecture
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device) #sends model to CPU/GPU

        self.model.eval() #sets model to eval mode (disables dropout, etc)

    def describe_image_with_interest(
        self,
        image: Image.Image,
        user_text: str,
        language: str = "en",
    ) -> VisionLLMResult:
        """
        Take a face image + user text and produce a descriptive text suitable
        as a caricature prompt.

        This sends a prompt + image to the vision-language model and returns
        the generated description.
        """

        # VERY IMPORTANT: Each VLM has its own prompt style.
        # For a general VLM, we use a simple instruction-style prompt.
        base_prompt = (
            "You are a helpful vision-language assistant. "
            "Look at the photo and describe the person in a friendly, "
            "caricature-oriented way. Focus on expressive but kind exaggerations "
            "of face and pose."
        )

        if user_text:
            base_prompt += f" Incorporate elements related to this hobby or interest: {user_text!r}."

        # Build processor inputs
        inputs = self.processor(
            text=base_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Generate output text
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True, #enable sampling to get more diverse outputs
                temperature=0.7, #controls randomness of sampling
            )

        # Decode the first sequence
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0] 

        return VisionLLMResult(description=text)
