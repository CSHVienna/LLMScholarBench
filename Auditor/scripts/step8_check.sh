#!/usr/bin/env bash
# Check that the number of files under each path matches the expected count.
# Usage: step8_check.sh <root_folder>
# Example: step8_check.sh ../results

set -euo pipefail

ROOT="${1:?Usage: $0 <root_folder> (e.g. ../results)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve ROOT relative to current dir (script dir)
if [[ "$ROOT" != /* ]]; then
  ROOT="$(cd "$ROOT" && pwd)"
fi

shopt -s nullglob
fail=0

check() {
  local path="$1"
  local expected="$2"
  local count
  local dir

  if [[ "$path" == */ ]]; then
    if [[ ! -d "$path" ]]; then
      printf "  FAIL %s (path doesn't exist)\n" "$path"
      fail=1
      return
    fi
    count=$(find "$path" -maxdepth 1 -type f 2>/dev/null | wc -l)
  else
    dir="$(dirname "$path")"
    if [[ ! -d "$dir" ]]; then
      printf "  FAIL %s (path doesn't exist)\n" "$path"
      fail=1
      return
    fi
    count=0
    for f in $path; do
      [[ -f "$f" ]] && (( count++ )) || true
    done
  fi

  if [[ "$count" -eq "$expected" ]]; then
    printf "  OK   %s => %s (expected %s)\n" "$path" "$count" "$expected"
  else
    printf "  FAIL %s => %s (expected %s)\n" "$path" "$count" "$expected"
    fail=1
  fi
}

echo "Root: $ROOT"
echo

# Summaries
echo "# Summaries"
check "$ROOT/temperature_analysis/summaries/" 24
check "$ROOT/interventions/summaries/" 24
echo

# Valid responses (validity)
echo "# Valid responses"
check "$ROOT/temperature_analysis/valid_responses/" 24
check "$ROOT/interventions/valid_responses/" 24
echo

# Factuality (author)
echo "# Factuality (author)"
check "$ROOT/temperature_analysis/factuality/*author*.csv" 24
check "$ROOT/interventions/factuality/*author*.csv" 24
echo

# Factuality (task)
echo "# Factuality (task)"
check "$ROOT/temperature_analysis/factuality/*field*.csv" 24
check "$ROOT/temperature_analysis/factuality/*epoch*.csv" 24
check "$ROOT/temperature_analysis/factuality/*seniority*.csv" 24
check "$ROOT/interventions/factuality/*field*.csv" 24
check "$ROOT/interventions/factuality/*epoch*.csv" 24
check "$ROOT/interventions/factuality/*seniority*.csv" 24
echo

# Similarities
echo "# Similarities"
check "$ROOT/temperature_analysis/similarities/*biased_top_k*.csv" 24
check "$ROOT/temperature_analysis/similarities/*top_k*.csv" 48
check "$ROOT/temperature_analysis/similarities/*field*.csv" 24
check "$ROOT/temperature_analysis/similarities/*epoch*.csv" 24
check "$ROOT/temperature_analysis/similarities/*seniority*.csv" 24
check "$ROOT/temperature_analysis/similarities/*twins*.csv" 24
check "$ROOT/interventions/similarities/*biased_top_k*.csv" 24
check "$ROOT/interventions/similarities/*top_k*.csv" 48
check "$ROOT/interventions/similarities/*field*.csv" 24
check "$ROOT/interventions/similarities/*epoch*.csv" 24
check "$ROOT/interventions/similarities/*seniority*.csv" 24
check "$ROOT/interventions/similarities/*twins*.csv" 24
echo

# Benchmarking
echo "# Benchmarking"
check "$ROOT/temperature_analysis/benchmarks/" 504
check "$ROOT/interventions/benchmarks/" 504
echo

# Refusals
echo "# Refusals"
check "$ROOT/refusals/*.csv" 2
check "$ROOT/refusals/*json" 1
check "$ROOT/refusals/*npy" 1
check "$ROOT/refusals/*jsonl" 1
echo

# Ground-truth
echo "# Ground-truth"
check "$ROOT/ground_truth/metadata/disciplines_author_demographics.csv" 1

shopt -u nullglob 2>/dev/null || true

echo
if [[ "$fail" -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed."
  exit 1
fi
