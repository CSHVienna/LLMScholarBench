#!/bin/bash
#SBATCH --job-name=llm_all_temps
#SBATCH --time=32:00:00
#SBATCH --output=logs/all_temps_%j.out
#SBATCH --error=logs/all_temps_%j.err

# Environment configuration
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# Temperature values
TEMPERATURES=(0 0.25 0.5 0.6 0.75 1 1.5 2)

echo "🌡️  Running experiments for all temperatures: ${TEMPERATURES[@]}"
echo "   Credentials: $LLMCALLER_CREDENTIALS"
echo "   Output: $LLMCALLER_OUTPUT"

cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run each temperature sequentially
for TEMP in "${TEMPERATURES[@]}"; do
    echo ""
    echo "========================================="
    echo "🌡️  Starting temperature: $TEMP"
    echo "========================================="

    # Run both providers in parallel for this temperature
    echo "Starting OpenRouter models with temp=$TEMP..."
    python3 main.py --all-models-smart --provider openrouter --batch-size 15 --temperature $TEMP > logs/openrouter_temp_${TEMP}.log 2>&1 &
    OPENROUTER_PID=$!

    echo "Starting Gemini models with temp=$TEMP..."
    python3 main.py --all-models-smart --provider gemini --temperature $TEMP > logs/gemini_temp_${TEMP}.log 2>&1 &
    GEMINI_PID=$!

    echo "Both providers started with temperature $TEMP. Waiting for completion..."

    # Wait for both to finish
    wait $OPENROUTER_PID
    OPENROUTER_STATUS=$?

    wait $GEMINI_PID
    GEMINI_STATUS=$?

    echo "✅ Temperature $TEMP completed"
    echo "OpenRouter exit status: $OPENROUTER_STATUS"
    echo "Gemini exit status: $GEMINI_STATUS"

    if [ $OPENROUTER_STATUS -ne 0 ] || [ $GEMINI_STATUS -ne 0 ]; then
        echo "❌ Temperature $TEMP had failures, but continuing..."
    fi

    # Small delay between temperatures
    sleep 10
done

echo ""
echo "🎉 All temperature experiments completed!"