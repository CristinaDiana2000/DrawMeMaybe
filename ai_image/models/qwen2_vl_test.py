from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

def main():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda"
    )

    
    image_path = "/home/rosrunner/drawmemby/DrawMeMaybe/ai_image/image_test/Amir.jpg"
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    
    prompt = "<image>\n Describe the user's hobbies/interests: I like coding, gaming, and watching animes."

    print("Preparing inputs...")
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to("cuda")

    print("Generating response...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150
    )

    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("\nQwen2-VL Response:\n", result)

if __name__ == "__main__":
    main()
