#!/bin/bash
# Final Experiments Runner - All paths configured in config/paths.json

set -e
cd /code/barolo/LLMScholar-Audits/LLMCaller

# Activate conda
eval "$(conda shell.bash hook)"
conda activate py312-llmcaller

# Run experiments (paths read from config/paths.json)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting experiments..."
python3 main.py --all-models-async --provider openrouter
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done!"
