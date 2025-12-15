#!/bin/bash

# Final Experiments Runner for LLMScholar-Audits
# Runs all OpenRouter models with optimized temperatures
# Server: mozart.csh.ac.at

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/code/barolo/LLMScholar-Audits/LLMCaller"
CREDENTIALS_DIR="/code/barolo/credentials"

# Output directory (override with environment variable or default)
OUTPUT_DIR="${LLMCALLER_OUTPUT:-/data/datasets/LLMScholar-Audits/LLMCaller/final_runs}"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Change to script directory
cd "${SCRIPT_DIR}"

# Activate conda (following Mozart guidelines)

source ~/.bashrc
conda activate llmscholar

# Export environment variables
export LLMCALLER_CREDENTIALS="${CREDENTIALS_DIR}"
export LLMCALLER_OUTPUT="${OUTPUT_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Log start
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
log "🚀 Starting final experiments"
log "   Output: ${OUTPUT_DIR}"
log "   Timestamp: ${RUN_TIMESTAMP}"

# Run experiments
python3 main.py --all-models-async --provider openrouter

# Log completion
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ Experiments completed successfully"
else
    log "❌ Experiments FAILED with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE
