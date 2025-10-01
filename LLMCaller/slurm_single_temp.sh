#!/bin/bash
#SBATCH --job-name=llm_temp_experiment
#SBATCH --time=04:00:00
#SBATCH --output=logs/temp_experiment_%j.out
#SBATCH --error=logs/temp_experiment_%j.err

# Get temperature from command line argument
TEMP=${1:-0}

echo "🌡️  Running experiments with temperature: $TEMP"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /path/to/your/LLMCaller  # YOU NEED TO UPDATE THIS PATH
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