from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

def main():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    print("Loading processor...")
    processor = LlavaNextProcessor.from_pretrained(model_id)

    print("Loading model...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda"
    )

    print("Loading image...")
    image_path = "/home/rosrunner/drawmemby/DrawMeMaybe/ai_image/image_test/Amir.jpg"
    image = Image.open(image_path)

    prompt = "Describe the user's hobbies/interests: I like coding, gaming, and watching animes."

    print("Generating LLaVA response...")
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=200
    )

    output_text = processor.decode(output_ids[0], skip_special_tokens=True)
    print("LLaVA Response:\n", output_text)

if __name__ == "__main__":
    main()
