#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/drawmemaybe/DrawMeMaybe"
cd "$PROJECT_DIR"

echo "=== 1) Starting Ollama service ==="
sudo systemctl start ollama

echo "=== 2) Running Agent (Ollama, prompt -> JSON) ==="
PYTHONPATH=. python3.12 sd_pipeline/run_agent_only.py

echo "=== 3) Stopping Ollama service ==="
sudo systemctl stop ollama

echo "=== 4) Running Stable Diffusion (reads JSON) ==="
PYTHONPATH=. python3.12 sd_pipeline/run_sd_only.py

echo "=== Done ==="
