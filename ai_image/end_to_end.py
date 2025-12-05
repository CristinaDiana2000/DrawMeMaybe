"""
End-to-end pipeline: Chat Input → Agent → SD → Output Image
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random
import logging

from ai_image.agent.caricature_agent import run_caricature_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaricatureSDPipeline:
    """End-to-end: Agent → SD Pipeline"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load SD model once (expensive operation)
        logger.info("Loading Stable Diffusion model...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to(self.device)
        logger.info("✓ Model loaded")
    
    def generate_caricature(
        self,
        input_image_path: str,
        user_text: str,
        output_path: str = "src/assets/caricature_result.png",
        strength: float = 0.7,
        guidance_scale: float = 7.5,
    ) -> dict:
        """
        Complete workflow:
        1. Load input image
        2. Run caricature agent (extract hobbies → generate prompt)
        3. Run SD img2img pipeline
        4. Save output
        5. Return metadata
        
        Args:
            input_image_path: Path to input image (e.g., from tablet photo)
            user_text: User's hobbies text (e.g., "I love climbing and painting")
            output_path: Where to save generated image
            strength: How much to transform the original (0-1)
            guidance_scale: How much to follow the prompt (1-20)
        
        Returns:
            dict with:
                - output_image_path
                - caricature_prompt
                - hobbies
                - exaggerations
                - generation_time_ms
        """
        
        logger.info("=" * 60)
        logger.info("🎨 Starting End-to-End Caricature Generation")
        logger.info("=" * 60)
        
        # STEP 1: Load input image
        logger.info(f"\n1.  Loading image from: {input_image_path}")
        init_image = Image.open(input_image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        logger.info(f"✓ Image loaded and resized to {init_image.size}")
        
        # STEP 2: Run caricature agent
        logger.info(f"\n2. Running Caricature Agent...")
        logger.info(f"   Input text: '{user_text}'")
        
        agent_result = run_caricature_agent(init_image, user_text)
        
        logger.info(f"✓ Agent completed in {agent_result['generation_time_ms']:.0f}ms")
        logger.info(f"   Hobbies: {agent_result['hobbies']}")
        logger.info(f"   Exaggerations: {agent_result['exaggerations']}")
        logger.info(f"   Title: {agent_result['title']}")
        
        prompt = agent_result['caricature_prompt']
        logger.info(f"   Generated prompt: {prompt[:100]}...")
        
        # STEP 3: Run SD Pipeline
        logger.info(f"\n3.  Running Stable Diffusion Pipeline...")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Strength: {strength} (how much to transform)")
        logger.info(f"   Guidance Scale: {guidance_scale} (how much to follow prompt)")
        
        seed = random.randint(0, 999999999)
        generator = torch.Generator(self.device).manual_seed(seed)
        logger.info(f"   Seed: {seed}")
        
        try:
            result_image = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
            logger.info("✓ SD Pipeline completed")
        except Exception as e:
            logger.error(f"✗ SD Pipeline failed: {e}")
            raise
        
        # STEP 4: Save output
        logger.info(f"\n4.  Saving result to: {output_path}")
        result_image.save(output_path)
        logger.info("✓ Image saved")
        
        # STEP 5: Return metadata
        logger.info(f"\n5.  Compilation complete!")
        logger.info("=" * 60)
        
        return {
            "output_image_path": output_path,
            "caricature_prompt": prompt,
            "title": agent_result['title'],
            "caption": agent_result['caption'],
            "hobbies": agent_result['hobbies'],
            "exaggerations": agent_result['exaggerations'],
            "agent_time_ms": agent_result['generation_time_ms'],
            "seed": seed,
        }


# ==============================
# Test / Usage Example
# ==============================

if __name__ == "__main__":
    import sys
    
    # Initialize pipeline
    pipeline = CaricatureSDPipeline()
    
    # Test with your image
    result = pipeline.generate_caricature(
        input_image_path="src/assets/whiteguy2.png",
        user_text="I love climbing and painting, also into photography",
        output_path="src/assets/caricature_result.png",
        strength=0.7,
        guidance_scale=7.5,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("🎉 COMPLETE RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key}: {value}")
