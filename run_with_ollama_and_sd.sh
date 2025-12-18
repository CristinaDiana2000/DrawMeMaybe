#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/drawmemaybe/drawmemaybe/DrawMeMaybe/backend"
cd "$PROJECT_DIR"
echo "PWD NOW: $(pwd)"

# Python soll das Projekt-Root als Modulpfad kennen
export PYTHONPATH="$PROJECT_DIR"

echo "=== 1) Starting Ollama service ==="
sudo systemctl start ollama

echo "=== 2) Running Agent (Ollama, prompt -> JSON) ==="
python3.12 ai_image/sd_pipeline/run_agent_only.py

echo "=== 3) Stopping Ollama service ==="
sudo systemctl stop ollama

echo "=== 4) Running Stable Diffusion (reads JSON) ==="
python3.12 ai_image/sd_pipeline/run_sd_only.py

echo "=== Done ==="
