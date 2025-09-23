#!/bin/bash
#SBATCH --job-name=llm_all_parallel
#SBATCH --time=02:00:00
#SBATCH --output=logs/both_parallel_%j.out
#SBATCH --error=logs/both_parallel_%j.err

# Environment configuration
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# Run both providers in parallel within single job
echo "🚀 Starting both OpenRouter and Gemini models in parallel"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run both in background, wait for both to complete
echo "Starting OpenRouter models..."
python3 main.py --all-models-smart --provider openrouter --batch-size 20 > logs/openrouter_inline.log 2>&1 &
OPENROUTER_PID=$!

echo "Starting Gemini models..."
python3 main.py --all-models-smart --provider gemini > logs/gemini_inline.log 2>&1 &
GEMINI_PID=$!

echo "Both providers started. Waiting for completion..."
echo "OpenRouter PID: $OPENROUTER_PID"
echo "Gemini PID: $GEMINI_PID"

# Wait for both to finish
wait $OPENROUTER_PID
OPENROUTER_STATUS=$?

wait $GEMINI_PID
GEMINI_STATUS=$?

echo "✅ Execution completed"
echo "OpenRouter exit status: $OPENROUTER_STATUS"
echo "Gemini exit status: $GEMINI_STATUS"

if [ $OPENROUTER_STATUS -eq 0 ] && [ $GEMINI_STATUS -eq 0 ]; then
    echo "🎉 All models completed successfully!"
    exit 0
else
    echo "❌ Some models failed"
    exit 1
fi