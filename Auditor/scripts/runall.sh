#!/usr/bin/env bash
# Run all pipeline steps (1–8) in order.
# Usage: ./runall.sh [LOG_DIR] [STEP8_ARG]
#   LOG_DIR   ; where to write logs (default: ../logs, i.e. Auditor/logs)
#   STEP8_ARG ; argument passed to step8_check.sh (default: ../results)
# Logs: ${LOG_DIR}/${TIMESTAMP}/${step}.log (timestamp = runall start time).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_BASE="${1:-$SCRIPT_DIR/../logs}"
STEP8_ARG="${2:-../results}"
# Single date call: epoch (for duration) and formatted timestamp (for log dir)
read -r START TIMESTAMP <<< "$(date +%s\ %Y%m%d_%H%M%S)"
LOG_DIR="${LOG_BASE}/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo "=== Running all steps ==="

for step in step1_validity step2_refusals step3_demographics step4_author_factuality step5_task_factuality step6_similarities step7_benchmarking; do
  LOG_FILE="${LOG_DIR}/${step}.log"
  echo "=== Running ${step}.sh -> ${LOG_FILE} ==="
  bash "./${step}.sh" 2>&1 | tee "$LOG_FILE"
done

LOG_FILE="${LOG_DIR}/step8_check.log"
echo "=== Running step8_check.sh -> ${LOG_FILE} ==="
bash "./step8_check.sh" "$STEP8_ARG" 2>&1 | tee "$LOG_FILE"

END=$(date +%s)
DURATION=$((END - START))
printf "=== All 8 steps completed ===\n"
printf "Total duration: %d seconds (%02d:%02d:%02d)\n" "$DURATION" "$((DURATION/3600))" "$(((DURATION%3600)/60))" "$((DURATION%60))"
