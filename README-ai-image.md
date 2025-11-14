# AI Image / Caricature Subproject

This folder contains a small, self-contained prototype for the
"Rob Ross Caricature Copilot Agent".

Goal:

- Input: face photo + short text about a hobby/interest
- Output: structured caricature concept:
  - title
  - caricature_prompt (for an image generator)
  - caption

Structure:

- `ai_image/models/vision_llm.py`:
  Wrapper for a vision-language model (dummy now, later BakLLaVA / LLaVA / Qwen-VL).
- `ai_image/agent/caricature_agent.py`:
  High-level agent logic.
- `ai_image/scripts/demo_caricature_cli.py`:
  Command line script to test the agent.

Next steps:

- Plug in a real vision-language model in `VisionLLM`.
- Later connect the `caricature_prompt` to Stable Diffusion / Flux.
