import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random

from ai_image.agent.caricature_agent import run_caricature_agent

image_path = "src/assets/whiteguy2.png"
user_text = "I love climbing, coding and cooking"

# Inputbild 
init_image = Image.open("src/assets/whiteguy2.png").convert("RGB")
init_image = init_image.resize((512, 512))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline laden 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  
).to(device)

#1.Step
print("Running agent...")
#Prompt from agent
agent_result = run_caricature_agent(image, user_text)
prompt = agent_result['caricature_prompt']

print(f"✓ Agent result:")
print(f"  Hobbies: {agent_result['hobbies']}")
print(f"  Exaggerations: {agent_result['exaggerations']}")
print(f"  Prompt: {prompt}")


# Prompt definieren 
#prompt = (
    #"funny caricature, huge head, tiny body, exaggerated features, cartoon style, bold outlines, simple shapes"
#)

#2.Step
# img2img Parameter
strength = 0.7           # wie stark das Original verändert wird
guidance_scale = 7.5      # wie stark das Modell dem Prompt folgt
seed = random.randint(0, 999999999)
generator = torch.Generator(device).manual_seed(seed)

# Inferenz
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

print("✓ SD Pipeline completed")


#3.Step

# Speichern und anzeigen
output_path = "src/assets/caricature_result.png"
result.save(output_path)
print(f"\n✅ Saved to: {output_path}")





