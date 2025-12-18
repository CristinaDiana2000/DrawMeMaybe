from pathlib import Path

# Base directory for this subproject (ai_image/)
BASE_DIR = Path(__file__).resolve().parent

# Where to store any test images, outputs, etc.
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Model configuration ---
# TO DO: change model name and device as needed--

# Later we’ll point this to the actual huggingface model id, e.g.:
# "bakllava" or a specific LLaVA / Qwen-VL model.
VISION_LLM_MODEL_NAME = "TODO-set-model-name"

# Device to run on: "cpu" for now on my laptop.
DEVICE = "cpu"