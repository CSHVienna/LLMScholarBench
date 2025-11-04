#!/bin/bash

# Temperature Experiment Runner - Runs all temps twice for OpenRouter only
# Usage: ./run_temperature_twice.sh

set -e  # Exit on error

# Configuration - EDIT THESE PATHS FOR YOUR SERVER
SCRIPT_DIR="/code/barolo/LLMScholar-Audits/LLMCaller"
OUTPUT_BASE="/data/datasets/LLMScholar-Audits/LLMCaller/data-temperature/twice_experiments"
CREDENTIALS_DIR="/code/barolo/credentials"

# Temperature values to test
TEMPERATURES=(0 0.25 0.5 0.75 1 1.5 2)

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Change to script directory
cd "${SCRIPT_DIR}"

# Activate conda environment (assuming you have a conda env for this project)
# If you need to create one first: conda create -n llmscholar python=3.10
# Then: conda activate llmscholar
log "Note: Make sure you have activated your conda environment before running this script"
log "Example: conda activate llmscholar"

# Export credentials path
export LLMCALLER_CREDENTIALS="${CREDENTIALS_DIR}"

# Main execution
log "🚀 Starting temperature experiments - 2 complete runs"
log "Output directory: ${OUTPUT_BASE}"
log "Temperatures: ${TEMPERATURES[*]}"
log "---"

# Run 1
log "========================================="
log "🔵 STARTING RUN 1"
log "========================================="
for temp in "${TEMPERATURES[@]}"; do
    # Format temperature as float with 1 decimal
    temp_formatted=$(printf "%.1f" "${temp}")
    log "🌡️  Run 1 - Temperature: ${temp_formatted}"

    # Structure: twice_experiments/run1/temperature_X.X/
    output_dir="${OUTPUT_BASE}/run1/temperature_${temp_formatted}"
    mkdir -p "${output_dir}"

    python3 main.py --all-models-async \
        --provider openrouter \
        --temperature "${temp}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/experiment.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✅ Run 1 - Temperature ${temp_formatted} completed successfully"
    else
        log "❌ Run 1 - Temperature ${temp_formatted} FAILED"
    fi

    log "⏳ Waiting 30 seconds before next temperature..."
    sleep 30
done

log "✅ Run 1 complete!"
log "---"

# Run 2
log "========================================="
log "🟢 STARTING RUN 2"
log "========================================="
for temp in "${TEMPERATURES[@]}"; do
    # Format temperature as float with 1 decimal
    temp_formatted=$(printf "%.1f" "${temp}")
    log "🌡️  Run 2 - Temperature: ${temp_formatted}"

    # Structure: twice_experiments/run2/temperature_X.X/
    output_dir="${OUTPUT_BASE}/run2/temperature_${temp_formatted}"
    mkdir -p "${output_dir}"

    python3 main.py --all-models-async \
        --provider openrouter \
        --temperature "${temp}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/experiment.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✅ Run 2 - Temperature ${temp_formatted} completed successfully"
    else
        log "❌ Run 2 - Temperature ${temp_formatted} FAILED"
    fi

    log "⏳ Waiting 30 seconds before next temperature..."
    sleep 30
done

log "========================================="
log "🎉 ALL EXPERIMENTS COMPLETE!"
log "========================================="
log "Total runs: 2"
log "Temperatures per run: ${#TEMPERATURES[@]}"
log "Models per temperature: 20 (OpenRouter only)"
log "Output location: ${OUTPUT_BASE}"
log "---"
log "Directory structure:"
log "  ${OUTPUT_BASE}/run1/temperature_0.0/config_<model>/run_<timestamp>/"
log "  ${OUTPUT_BASE}/run2/temperature_0.0/config_<model>/run_<timestamp>/"
log "---"
log "To check results:"
log "  ls -la ${OUTPUT_BASE}/run1/"
log "  ls -la ${OUTPUT_BASE}/run2/"
