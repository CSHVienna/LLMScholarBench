#!/bin/bash

# Temperature Experiment Runner
# This script runs LLM experiments at different temperature values
# for both OpenRouter and Gemini providers

# Configuration
export LLMCALLER_CREDENTIALS="/home/barolo/credentials"
export LLMCALLER_OUTPUT="/raid5pool/tank/barolo/llmscholar/experiments"
VENV_PATH="/home/barolo/.venv/bin/activate"
SCRIPT_DIR="/home/barolo/LLMCaller"

# Temperature values to test
TEMPERATURES=(0 0.25 0.5 0.75 1 1.5 2)

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run experiment for a specific provider and temperature
run_experiment() {
    local provider=$1
    local temperature=$2
    local experiment_id="temp_${temperature}_${provider}_$(date '+%Y%m%d_%H%M%S')"

    log "🌡️  Starting ${provider} experiment at temperature ${temperature}"

    # Activate virtual environment and run experiment
    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}"

    # Create timestamped output directory
    output_dir="${LLMCALLER_OUTPUT}/${experiment_id}"
    mkdir -p "${output_dir}"

    # Run the experiment with error handling
    if python3 main.py --all-models-smart --provider "${provider}" --temperature "${temperature}" --output-dir "${output_dir}" 2>&1 | tee "${output_dir}/experiment.log"; then
        log "✅ ${provider} experiment at temperature ${temperature} completed successfully"
        echo "${experiment_id}" >> "${LLMCALLER_OUTPUT}/completed_experiments.log"
    else
        log "❌ ${provider} experiment at temperature ${temperature} failed"
        echo "${experiment_id} FAILED" >> "${LLMCALLER_OUTPUT}/failed_experiments.log"
    fi
}

# Function to run both providers in parallel for a given temperature
run_parallel_experiments() {
    local temperature=$1

    log "🚀 Starting parallel experiments for temperature ${temperature}"

    # Run OpenRouter and Gemini in parallel
    run_experiment "openrouter" "${temperature}" &
    local openrouter_pid=$!

    run_experiment "gemini" "${temperature}" &
    local gemini_pid=$!

    # Wait for both to complete
    wait $openrouter_pid
    local openrouter_exit=$?

    wait $gemini_pid
    local gemini_exit=$?

    if [ $openrouter_exit -eq 0 ] && [ $gemini_exit -eq 0 ]; then
        log "✅ Both providers completed successfully for temperature ${temperature}"
    else
        log "⚠️  One or both providers failed for temperature ${temperature} (OpenRouter: $openrouter_exit, Gemini: $gemini_exit)"
    fi
}

# Function to run single provider experiments
run_single_provider() {
    local provider=$1
    local temperature=$2

    if [ -z "$temperature" ]; then
        log "🔄 Running all temperatures for ${provider}"
        for temp in "${TEMPERATURES[@]}"; do
            run_experiment "${provider}" "${temp}"
            # Small delay between experiments
            sleep 10
        done
    else
        log "🎯 Running single temperature ${temperature} for ${provider}"
        run_experiment "${provider}" "${temperature}"
    fi
}

# Main execution logic
main() {
    local mode="$1"
    local provider="$2"
    local temperature="$3"

    log "🚀 Starting temperature experiments"
    log "Mode: ${mode:-parallel}, Provider: ${provider:-both}, Temperature: ${temperature:-all}"

    # Create log directories
    mkdir -p "${LLMCALLER_OUTPUT}"

    case "$mode" in
        "single")
            if [ -n "$provider" ]; then
                run_single_provider "$provider" "$temperature"
            else
                echo "Error: Provider required for single mode"
                echo "Usage: $0 single [openrouter|gemini] [temperature]"
                exit 1
            fi
            ;;
        "parallel"|"")
            if [ -n "$temperature" ]; then
                run_parallel_experiments "$temperature"
            else
                log "🔄 Running all temperatures in parallel mode"
                for temp in "${TEMPERATURES[@]}"; do
                    run_parallel_experiments "$temp"
                    # Delay between temperature sets to avoid overwhelming the system
                    log "⏳ Waiting 60 seconds before next temperature set..."
                    sleep 60
                done
            fi
            ;;
        *)
            echo "Usage: $0 [parallel|single] [provider] [temperature]"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all temperatures, both providers in parallel"
            echo "  $0 parallel 0.5                      # Run temperature 0.5 for both providers in parallel"
            echo "  $0 single openrouter                 # Run all temperatures for OpenRouter only"
            echo "  $0 single gemini 1.0                 # Run temperature 1.0 for Gemini only"
            echo ""
            echo "Available temperatures: ${TEMPERATURES[*]}"
            exit 1
            ;;
    esac

    log "🎉 Temperature experiments completed"
}

# Handle script interruption gracefully
trap 'log "⚠️  Script interrupted"; exit 130' INT TERM

# Run main function with all arguments
main "$@"