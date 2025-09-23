#!/bin/bash
#SBATCH --job-name=llm_openrouter
#SBATCH --time=02:00:00
#SBATCH --output=logs/openrouter_%j.out
#SBATCH --error=logs/openrouter_%j.err

# Environment configuration
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# OpenRouter models with smart queue system
echo "🚀 Starting OpenRouter models with smart queue"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate

python3 main.py --all-models-smart --provider openrouter --batch-size 20

echo "✅ OpenRouter execution completed"