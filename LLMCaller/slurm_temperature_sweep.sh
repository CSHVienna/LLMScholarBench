#!/bin/bash
#SBATCH --job-name=llm_temp_sweep
#SBATCH --time=16:00:00
#SBATCH --output=logs/temp_sweep_%j.out
#SBATCH --error=logs/temp_sweep_%j.err
#SBATCH --array=0-7

# Environment configuration
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# Temperature values array
TEMPERATURES=(0 0.25 0.5 0.6 0.75 1 1.5 2)
TEMP=${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}

echo "🌡️  Running experiments with temperature: $TEMP"
echo "   Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run both providers in parallel with specific temperature
echo "Starting OpenRouter models with temp=$TEMP..."
python3 main.py --all-models-smart --provider openrouter --batch-size 15 --temperature $TEMP > logs/openrouter_temp_${TEMP}_${SLURM_JOB_ID}.log 2>&1 &
OPENROUTER_PID=$!

echo "Starting Gemini models with temp=$TEMP..."
python3 main.py --all-models-smart --provider gemini --temperature $TEMP > logs/gemini_temp_${TEMP}_${SLURM_JOB_ID}.log 2>&1 &
GEMINI_PID=$!

echo "Both providers started with temperature $TEMP. Waiting for completion..."
echo "OpenRouter PID: $OPENROUTER_PID"
echo "Gemini PID: $GEMINI_PID"

# Wait for both to finish
wait $OPENROUTER_PID
OPENROUTER_STATUS=$?

wait $GEMINI_PID
GEMINI_STATUS=$?

echo "✅ Temperature $TEMP experiments completed"
echo "OpenRouter exit status: $OPENROUTER_STATUS"
echo "Gemini exit status: $GEMINI_STATUS"

if [ $OPENROUTER_STATUS -eq 0 ] && [ $GEMINI_STATUS -eq 0 ]; then
    echo "🎉 Temperature $TEMP completed successfully!"
    exit 0
else
    echo "❌ Temperature $TEMP had failures"
    exit 1
fi