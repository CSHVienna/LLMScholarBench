#!/bin/bash
#SBATCH --job-name=llm_gemini
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemini_%j.out
#SBATCH --error=logs/gemini_%j.err

# Environment configuration
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# Gemini models with concurrent execution (no rate limits)
echo "🧠 Starting Gemini models with concurrent execution"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate

python3 main.py --all-models-smart --provider gemini

echo "✅ Gemini execution completed"