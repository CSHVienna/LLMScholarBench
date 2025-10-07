#!/bin/bash

# Cron Job Setup for Temperature Experiments
# This script sets up automated cron jobs to run temperature experiments

SCRIPT_DIR="/home/barolo/LLMCaller"
EXPERIMENT_SCRIPT="${SCRIPT_DIR}/run_temperature_experiments.sh"
CRON_LOG_DIR="/raid5pool/tank/barolo/llmscholar/experiments/cron_logs"
CRON_FILE="/tmp/temperature_experiments_cron"

# Create log directory
mkdir -p "${CRON_LOG_DIR}"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create cron entries
create_cron_entries() {
    cat > "${CRON_FILE}" << 'EOF'
# Temperature Experiments Cron Jobs
# Format: minute hour day month weekday command
# Runs Oct 6-14, 2025 only

# Daily 8:00 AM - Temperature 0.5 (both providers parallel) - Oct 6-14
0 8 6-14 10 * [ "$(date +\%Y)" = "2025" ] && /home/barolo/LLMCaller/run_temperature_experiments.sh parallel "" 0.5 >> /raid5pool/tank/barolo/llmscholar/experiments/cron_logs/temp_0.5.log 2>&1

# Daily 8:00 PM - Temperature 0.5 (both providers parallel) - Oct 6-14
0 20 6-14 10 * [ "$(date +\%Y)" = "2025" ] && /home/barolo/LLMCaller/run_temperature_experiments.sh parallel "" 0.5 >> /raid5pool/tank/barolo/llmscholar/experiments/cron_logs/temp_0.5.log 2>&1

EOF
}

# Function to install cron jobs
install_cron_jobs() {
    log "📅 Installing cron jobs for temperature experiments"

    # Backup existing crontab
    crontab -l > "${CRON_LOG_DIR}/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "# No existing crontab" > "${CRON_LOG_DIR}/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"

    # Create new cron entries
    create_cron_entries

    # Install the new cron jobs (append to existing crontab)
    (crontab -l 2>/dev/null; echo ""; cat "${CRON_FILE}") | crontab -

    log "✅ Cron jobs installed successfully"
    log "📋 Current cron jobs:"
    crontab -l | grep -A 20 "Temperature Experiments"
}

# Function to remove cron jobs
remove_cron_jobs() {
    log "🗑️  Removing temperature experiment cron jobs"

    # Remove lines containing the experiment script
    crontab -l 2>/dev/null | grep -v "/home/barolo/LLMCaller/run_temperature_experiments.sh" | crontab -

    log "✅ Temperature experiment cron jobs removed"
}

# Function to show current cron jobs
show_cron_jobs() {
    log "📋 Current temperature experiment cron jobs:"
    crontab -l 2>/dev/null | grep "/home/barolo/LLMCaller/run_temperature_experiments.sh" || echo "No temperature experiment cron jobs found"
}

# Function to test a single cron job
test_cron_job() {
    local temperature="$1"

    if [ -z "$temperature" ]; then
        echo "Usage: $0 test [temperature]"
        echo "Example: $0 test 0.5"
        exit 1
    fi

    log "🧪 Testing temperature experiment for temperature ${temperature}"
    "${EXPERIMENT_SCRIPT}" parallel "" "${temperature}"
}

# Main execution
main() {
    local action="$1"

    case "$action" in
        "install")
            install_cron_jobs
            ;;
        "remove")
            remove_cron_jobs
            ;;
        "show"|"list")
            show_cron_jobs
            ;;
        "test")
            test_cron_job "$2"
            ;;
        *)
            echo "Usage: $0 [install|remove|show|test]"
            echo ""
            echo "Commands:"
            echo "  install  - Install cron jobs for automated temperature experiments"
            echo "  remove   - Remove temperature experiment cron jobs"
            echo "  show     - Show current temperature experiment cron jobs"
            echo "  test     - Test a single temperature experiment"
            echo ""
            echo "Schedule:"
            echo "  Daily 8:00 AM  - Temperature 0.5 (all models, both providers) - Oct 6-14, 2025"
            echo "  Daily 8:00 PM  - Temperature 0.5 (all models, both providers) - Oct 6-14, 2025"
            echo ""
            echo "Logs are saved to: ${CRON_LOG_DIR}"
            exit 1
            ;;
    esac
}

# Cleanup temporary file on exit
trap 'rm -f "${CRON_FILE}"' EXIT

# Run main function
main "$@"