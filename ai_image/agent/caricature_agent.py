"""
Caricature Agent v2 - LLaMA-powered with Ollama

Workflow:
1. Extract hobbies from user chat input (LLM)
2. Generate SD prompt based on hobbies + image (LLM)
   - With random exaggerations of body features
3. Return title, prompt, caption for pipeline

No face description needed - just hobbies → creative prompt.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from PIL import Image
import json #json parsing
import time
import logging #debugging output
import re #Regex für Text-Parsing
import random #für exaggerations
from ollama import Client 

# Configure logging
logging.basicConfig(
    level=logging.INFO, #zeigt INFO+ messages
    format='%(asctime)s - %(levelname)s - %(message)s' #Format: Zeit - Level - Message
)
logger = logging.getLogger(__name__)


# ==============================
# Data Structures
# ==============================

@dataclass
class CaricatureConcept:
    """Output structure for the agent"""
    title: str
    caricature_prompt: str
    caption: str
    hobbies: List[str]
    exaggerations: List[str]
    generation_time_ms: float

    def to_dict(self) -> Dict:          #konvertiert dataclass zu dictionary für json
        return {
            "title": self.title,
            "caricature_prompt": self.caricature_prompt,
            "caption": self.caption,
            "hobbies": self.hobbies,
            "exaggerations": self.exaggerations,
            "generation_time_ms": self.generation_time_ms,
        }


# ==============================
# Ollama Client Setup
# ==============================

class OllamaAgent:
    """Wrapper around Ollama client with error handling"""
    
    def __init__(self, model: str = "mistral:7b-instruct-q4_K_M", 
                 host: str = "http://localhost:11434"):
        self.model = model          #model name
        self.host = host            #ollama host url
        self.client = Client(host=host)     #erstellt ollama client
        self._check_connection()        #prüft ob es läuft
    
    def _check_connection(self) -> bool:
        """Verify Ollama is running"""
        try:
            # Try a simple ping
            self.client.generate(
                model=self.model,
                prompt="test",
                stream=False,
            )
            logger.info(f"✓ Connected to Ollama at {self.host}")
            return True
        except Exception as e:
            logger.error(f"✗ Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise       #wirft fehlermeldung, damit agent nicht weiterläuft mit kaputtem setup 
    
    @staticmethod
    def _get_random_exaggerations() -> List[str]:
        """
        Return 2-3 random body features to exaggerate.
        Adds variety to each caricature generation.
        """
        features = [
            "eyes",
            "nose",
            "ears",
            "mouth",
            "chin",
            "cheeks",
            "hair",
            "eyebrows",
            "forehead",
            "jaw",
            "teeth",
            "fingers",
            "hands",
            "arms",
        ]
        # Pick 2-3 random features
        return random.sample(features, k=random.randint(2, 3))
    
    def extract_hobbies(self, user_text: str) -> List[str]:
        """
        Extract hobbies from user input using LLM.
        Robust parsing that handles Mistral's JSON inconsistencies.
        """
        if not user_text or len(user_text.strip()) < 3:
            logger.warning("Empty or too short user input")
            return []
        
        prompt = f"""Extract hobbies from this text. Return ONLY a JSON array.

Text: "{user_text}"

Return format: ["hobby1", "hobby2", "hobby3"]

Response:"""
        
        try:
            #sendet prompt an Ollama/Mistrial
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            
            response_text = response['response'].strip()
            logger.info(f"LLM response: {response_text[:200]}")
            
            # Try 1: Direct JSON parsing
            try:
                hobbies = json.loads(response_text)
                if isinstance(hobbies, list):
                    hobbies = [h.strip() for h in hobbies if h.strip()]
                    logger.info(f"Extracted hobbies (direct JSON): {hobbies}")
                    return hobbies
            except json.JSONDecodeError:
                pass #weitemachen zu Try 2
            
            # Try 2: Extract JSON array from text
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                try:
                    hobbies = json.loads(json_match.group())
                    if isinstance(hobbies, list):
                        hobbies = [h.strip() for h in hobbies if h.strip()]
                        logger.info(f"Extracted hobbies (from text): {hobbies}")
                        return hobbies
                except json.JSONDecodeError:
                    pass #weitermachen zu Try 3
            
            # Try 3: Simple split on commas if text looks like a list
            if ',' in response_text or 'and' in response_text.lower():
                # Split by comma or "and"
                hobbies = re.split(r',|\sand\s', response_text)
                hobbies = [h.strip() for h in hobbies if h.strip() and len(h.strip()) > 2]
                if hobbies:
                    logger.info(f"Extracted hobbies (fallback split): {hobbies}")
                    return hobbies
            
            logger.warning(f"Could not parse hobbies from: {response_text}")
            return []
            
        except Exception as e:
            logger.error(f"Hobby extraction failed: {e}")
            return []
        
    #Generate Prompt
    
    def generate_prompt(self, hobbies: List[str], image: Image.Image, 
                       exaggerations: List[str]) -> str:
        """
        Generate SD prompt based on hobbies, image, and random exaggerations.
        The prompt should work well for:
        1. Caricature generation
        2. Robotic arm drawing (clean lines, contours)
        3. Random variety in exaggerated features
        """
        #erstellt komma-separierte Strings
        hobbies_str = ", ".join(hobbies) if hobbies else "general creative interests" #z.B "climbing","painting"
        exaggerations_str = ", ".join(exaggerations)  #z.B "eyes". "nose"
        
        prompt = f"""Generate a Stable Diffusion prompt for a caricature. The prompt should:
1. Create a playful, exaggerated caricature style
2. Incorporate the person's hobbies/interests visually
3. EXAGGERATE these features dramatically: {exaggerations_str}
4. Be suitable for line art and robotic arm drawing (clean outlines, minimal shading)
5. Include style hints: "clean line art, bold contours, minimal detail"
6. Be 1-2 sentences max

Person's interests: {hobbies_str}

Generate ONLY the prompt, nothing else:"""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            
            sd_prompt = response['response'].strip()
            logger.info(f"Generated SD prompt (exaggerating {exaggerations_str}): {sd_prompt[:100]}...")
            return sd_prompt
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            # Fallback prompt
            return f"A playful caricature with interests in {hobbies_str}, exaggerating {exaggerations_str}. Style: clean line art, bold contours."


# ==============================
# Main Agent
# ==============================

class CaricatureAgent:
    """Main agent orchestrator"""
    
    def __init__(self, model: str = "mistral:7b-instruct-q4_K_M"):
        self.llm = OllamaAgent(model=model) #erstellt Ollama Wrapper
    
    def run(self, image: Image.Image, user_text: str) -> CaricatureConcept:
        """
        Main entrypoint.
        
        Args:
            image: PIL Image (not used for description, but passed for context)
            user_text: User's chat input with hobbies
        
        Returns:
            CaricatureConcept with title, prompt, caption, hobbies, exaggerations
        """
        start_time = time.time()
        
        # Step 1: Extract hobbies
        #Input: "I love climbing and painting"
        #Output: ['climbing', 'painting']
        hobbies = self.llm.extract_hobbies(user_text)
        
        # Step 2: Get random exaggerations
        #Output: ['eyes', 'nose']
        exaggerations = self.llm._get_random_exaggerations()
        
        # Step 3: Generate SD prompt (with exaggerations)
        #Output: "Create a playful caricature ..."
        sd_prompt = self.llm.generate_prompt(hobbies, image, exaggerations)
        
        # Step 4: Build concept
        title = self._build_title(hobbies)  #Output: "The Climbing Caricature"
        caption = self._build_caption(hobbies, exaggerations) #Output: "A caricature inspired by ... with exaggerated ..."
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        concept = CaricatureConcept(
            title=title,
            caricature_prompt=sd_prompt,
            caption=caption,
            hobbies=hobbies,
            exaggerations=exaggerations,
            generation_time_ms=elapsed_ms,
        )
        
        logger.info(f"Agent completed in {elapsed_ms:.0f}ms")
        return concept
    
    @staticmethod
    def _build_title(hobbies: List[str]) -> str:
        """Build a title from hobbies"""
        if hobbies:
            main = hobbies[0].title()
            return f"The {main} Caricature"
        return "Personal Caricature"
    
    @staticmethod
    def _build_caption(hobbies: List[str], exaggerations: List[str]) -> str:
        """Build a caption from hobbies and exaggerations"""
        if hobbies:
            hobbies_str = ", ".join(hobbies)
            exag_str = ", ".join(exaggerations)
            return f"A caricature inspired by your interests in {hobbies_str}, with exaggerated {exag_str}."
        return "A playful caricature based on your personality."


# ==============================
# Public API (backward compatible)
# ==============================

_agent: Optional[CaricatureAgent] = None    #global singleton instance

def _get_agent() -> CaricatureAgent:
    """Lazy-load agent singleton"""
    global _agent
    if _agent is None:
        _agent = CaricatureAgent()  #erstelle beim ersten aufruf
    return _agent #nächste aufrufe geben selbe Instanz zurück

def run_caricature_agent(image: Image.Image, user_text: str) -> Dict:
    """
    Main entrypoint for your backend.
    
    Input:
        - image (PIL.Image)
        - user_text (str): User's chat input with hobbies
    
    Output:
        dict with:
            - title
            - caption
            - caricature_prompt
            - hobbies (list)
            - exaggerations (list) - random body features to exaggerate
            - generation_time_ms
    """
    agent = _get_agent()    #hole oder erstelle agent
    concept = agent.run(image, user_text)   #run workflow
    return concept.to_dict()    #konvertiert zu dictionary


# ==============================
# Testing / Debug
# ==============================

if __name__ == "__main__":
    # Quick test
    from PIL import Image
    import sys
    
    # Create a dummy image for testing
    test_image = Image.new('RGB', (512, 512), color='white')
    test_text = "I love coding and reading, also into animes"
    
    print("\n🎨 Caricature Agent Test\n")
    print(f"Input text: {test_text}")
    print(f"Image size: {test_image.size}")
    
    try:
        result = run_caricature_agent(test_image, test_text)
        print("\n✓ Success!\n")
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        sys.exit(1)