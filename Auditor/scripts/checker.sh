#!/usr/bin/env bash
# Show counts of result CSVs by model and by kind/metric/task depending on folder type.
# Supports: benchmarks, factuality, similarities, summaries, valid_responses.
# Usage: checker.sh <path>
# Example: checker.sh ../results/benchmarking   or   checker.sh ../results/factuality

set -euo pipefail

PATH_ARG="${1:?Usage: $0 <path> (e.g. ../results/benchmarks, ../results/factuality, ...)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve path
if [[ "$PATH_ARG" != /* ]]; then
  DIR="$(cd "$PATH_ARG" && pwd)"
else
  DIR="$PATH_ARG"
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: not a directory: $DIR" >&2
  exit 1
fi

BASENAME="$(basename "$DIR")"

# -----------------------------------------------------------------------------
# Benchmarks: per_attempt_<model>_<metric>.csv and temperature_per_attempt_*.csv
# -----------------------------------------------------------------------------
run_benchmarks() {
  local METRICS=(
    connectedness_norm_entropy
    connectedness_ncomponents
    factuality_seniority
    diversity_prominence_cit
    diversity_prominence_pub
    parity_prominence_cit
    parity_prominence_pub
    factuality_author
    factuality_epoch
    factuality_field
    connectedness_density
    similarity_pca
    diversity_ethnicity
    diversity_gender
    parity_ethnicity
    parity_gender
    connectedness
    refusal_pct
    validity_pct
    duplicates
    consistency
  )
  shopt -s nullglob
  FILES=()
  for f in "$DIR"/*per_attempt_*.csv; do
    FILES+=("$f")
  done
  shopt -u nullglob 2>/dev/null || true

  strip_prefix() {
    local base="$1"
    if [[ "$base" == temperature_per_attempt_* ]]; then
      echo "${base#temperature_per_attempt_}"
    elif [[ "$base" == per_attempt_* ]]; then
      echo "${base#per_attempt_}"
    else
      echo "$base"
    fi
  }

  parse_stem() {
    local stem="$1"
    local m
    for m in "${METRICS[@]}"; do
      if [[ "$stem" == *_"$m" ]]; then
        local model="${stem%_"$m"}"
        echo "$model	$m"
        return 0
      fi
    done
    echo "	$stem"
    return 1
  }

  declare -A BY_MODEL
  declare -A BY_METRIC
  UNPARSED=()
  for f in "${FILES[@]}"; do
    base=$(basename "$f" .csv)
    stem=$(strip_prefix "$base")
    pair=$(parse_stem "$stem") || true
    model="${pair%%	*}"
    metric="${pair#*	}"
    if [[ -n "$model" ]]; then
      BY_MODEL["$model"]=$((${BY_MODEL["$model"]:-0} + 1))
      BY_METRIC["$metric"]=$((${BY_METRIC["$metric"]:-0} + 1))
    else
      UNPARSED+=("$base")
    fi
  done

  EXPECTED_METRICS=${#METRICS[@]}
  NUM_MODELS=${#BY_MODEL[@]}
  EXPECTED_PER_METRIC=${NUM_MODELS:-0}

  echo "Path: $DIR"
  echo "Pattern: temperature_per_attempt_*.csv and/or per_attempt_*.csv"
  echo "Total files: ${#FILES[@]}"
  echo ""
  echo "=== By model (expected $EXPECTED_METRICS each) ==="
  for model in $(printf '%s\n' "${!BY_MODEL[@]}" | sort); do
    c=${BY_MODEL[$model]}
    if [[ "$c" -ne "$EXPECTED_METRICS" ]]; then
      echo "  $c	$model  <-- expected $EXPECTED_METRICS"
    else
      echo "  $c	$model"
    fi
  done
  echo ""
  echo "=== By metric (expected $EXPECTED_PER_METRIC each) ==="
  for metric in $(printf '%s\n' "${!BY_METRIC[@]}" | sort); do
    c=${BY_METRIC[$metric]}
    if [[ "$c" -ne "$EXPECTED_PER_METRIC" ]]; then
      echo "  $c	$metric  <-- expected $EXPECTED_PER_METRIC"
    else
      echo "  $c	$metric"
    fi
  done
  if [[ ${#UNPARSED[@]} -gt 0 ]]; then
    echo ""
    echo "=== Unparsed filenames ==="
    printf '%s\n' "${UNPARSED[@]}"
  fi
}

# -----------------------------------------------------------------------------
# Factuality: <model>_author.csv, <model>_epoch.csv, <model>_field.csv, <model>_seniority.csv
# 4 per model; expected per kind = number of models
# -----------------------------------------------------------------------------
run_factuality() {
  local KINDS=(author epoch field seniority)
  shopt -s nullglob
  FILES=()
  for k in "${KINDS[@]}"; do
    for f in "$DIR"/*_"$k".csv; do
      FILES+=("$f")
    done
  done
  shopt -u nullglob 2>/dev/null || true

  declare -A BY_MODEL
  declare -A BY_KIND
  for f in "${FILES[@]}"; do
    base=$(basename "$f" .csv)
    for k in "${KINDS[@]}"; do
      if [[ "$base" == *_"$k" ]]; then
        model="${base%_"$k"}"
        BY_MODEL["$model"]=$((${BY_MODEL["$model"]:-0} + 1))
        BY_KIND["$k"]=$((${BY_KIND["$k"]:-0} + 1))
        break
      fi
    done
  done

  EXPECTED_PER_MODEL=4
  NUM_MODELS=${#BY_MODEL[@]}
  EXPECTED_PER_KIND=${NUM_MODELS:-0}

  echo "Path: $DIR"
  echo "Pattern: *_author.csv, *_epoch.csv, *_field.csv, *_seniority.csv"
  echo "Total files: ${#FILES[@]}"
  echo ""
  echo "=== By model (expected $EXPECTED_PER_MODEL each) ==="
  for model in $(printf '%s\n' "${!BY_MODEL[@]}" | sort); do
    c=${BY_MODEL[$model]}
    if [[ "$c" -ne "$EXPECTED_PER_MODEL" ]]; then
      echo "  $c	$model  <-- expected $EXPECTED_PER_MODEL"
    else
      echo "  $c	$model"
    fi
  done
  echo ""
  echo "=== By kind (expected $EXPECTED_PER_KIND each) ==="
  for kind in $(printf '%s\n' "${!BY_KIND[@]}" | sort); do
    c=${BY_KIND[$kind]}
    if [[ "$c" -ne "$EXPECTED_PER_KIND" ]]; then
      echo "  $c	$kind  <-- expected $EXPECTED_PER_KIND"
    else
      echo "  $c	$kind"
    fi
  done
}

# -----------------------------------------------------------------------------
# Similarities: <model>_epoch.csv, _field.csv, _seniority.csv, _top_k.csv, _twins.csv, _biased_top_k.csv
# 6 per model; expected per task = number of models.
# Order tasks by length descending so "biased_top_k" is matched before "top_k".
# -----------------------------------------------------------------------------
run_similarities() {
  local TASKS=(biased_top_k epoch field seniority top_k twins)
  shopt -s nullglob
  # Deduplicate: *_top_k.csv matches both model_top_k.csv and model_biased_top_k.csv,
  # so collect paths in an associative array so each file is only processed once.
  declare -A FILES_SET
  for t in "${TASKS[@]}"; do
    for f in "$DIR"/*_"$t".csv; do
      FILES_SET["$f"]=1
    done
  done
  FILES=("${!FILES_SET[@]}")
  shopt -u nullglob 2>/dev/null || true

  declare -A BY_MODEL
  declare -A BY_TASK
  for f in "${FILES[@]}"; do
    base=$(basename "$f" .csv)
    for t in "${TASKS[@]}"; do
      if [[ "$base" == *_"$t" ]]; then
        model="${base%_"$t"}"
        BY_MODEL["$model"]=$((${BY_MODEL["$model"]:-0} + 1))
        BY_TASK["$t"]=$((${BY_TASK["$t"]:-0} + 1))
        break
      fi
    done
  done

  EXPECTED_PER_MODEL=6
  NUM_MODELS=${#BY_MODEL[@]}
  EXPECTED_PER_TASK=${NUM_MODELS:-0}

  echo "Path: $DIR"
  echo "Pattern: *_epoch.csv, *_field.csv, *_seniority.csv, *_top_k.csv, *_twins.csv, *_biased_top_k.csv"
  echo "Total files: ${#FILES[@]}"
  echo ""
  echo "=== By model (expected $EXPECTED_PER_MODEL each) ==="
  for model in $(printf '%s\n' "${!BY_MODEL[@]}" | sort); do
    c=${BY_MODEL[$model]}
    if [[ "$c" -ne "$EXPECTED_PER_MODEL" ]]; then
      echo "  $c	$model  <-- expected $EXPECTED_PER_MODEL"
    else
      echo "  $c	$model"
    fi
  done
  echo ""
  echo "=== By task (expected $EXPECTED_PER_TASK each) ==="
  for task in $(printf '%s\n' "${!BY_TASK[@]}" | sort); do
    c=${BY_TASK[$task]}
    if [[ "$c" -ne "$EXPECTED_PER_TASK" ]]; then
      echo "  $c	$task  <-- expected $EXPECTED_PER_TASK"
    else
      echo "  $c	$task"
    fi
  done
}

# -----------------------------------------------------------------------------
# Summaries: experiments_<model>.csv — 1 per model
# -----------------------------------------------------------------------------
run_summaries() {
  shopt -s nullglob
  FILES=()
  for f in "$DIR"/experiments_*.csv; do
    FILES+=("$f")
  done
  shopt -u nullglob 2>/dev/null || true

  declare -A BY_MODEL
  for f in "${FILES[@]}"; do
    base=$(basename "$f" .csv)
    model="${base#experiments_}"
    BY_MODEL["$model"]=$((${BY_MODEL["$model"]:-0} + 1))
  done

  EXPECTED_PER_MODEL=1
  echo "Path: $DIR"
  echo "Pattern: experiments_*.csv"
  echo "Total files: ${#FILES[@]}"
  echo ""
  echo "=== By model (expected $EXPECTED_PER_MODEL each) ==="
  for model in $(printf '%s\n' "${!BY_MODEL[@]}" | sort); do
    c=${BY_MODEL[$model]}
    if [[ "$c" -ne "$EXPECTED_PER_MODEL" ]]; then
      echo "  $c	$model  <-- expected $EXPECTED_PER_MODEL"
    else
      echo "  $c	$model"
    fi
  done
}

# -----------------------------------------------------------------------------
# Valid_responses: <model>.csv — 1 per model (any .csv whose basename has no _kind suffix)
# -----------------------------------------------------------------------------
run_valid_responses() {
  shopt -s nullglob
  FILES=()
  for f in "$DIR"/*.csv; do
    base=$(basename "$f" .csv)
    # Exclude experiments_* (summaries) and *_author, *_epoch, etc. (factuality-style)
    if [[ "$base" != experiments_* && "$base" != *_author && "$base" != *_epoch && "$base" != *_field && "$base" != *_seniority ]]; then
      FILES+=("$f")
    fi
  done
  shopt -u nullglob 2>/dev/null || true

  declare -A BY_MODEL
  for f in "${FILES[@]}"; do
    base=$(basename "$f" .csv)
    BY_MODEL["$base"]=$((${BY_MODEL["$base"]:-0} + 1))
  done

  EXPECTED_PER_MODEL=1
  echo "Path: $DIR"
  echo "Pattern: <model>.csv (one CSV per model)"
  echo "Total files: ${#FILES[@]}"
  echo ""
  echo "=== By model (expected $EXPECTED_PER_MODEL each) ==="
  for model in $(printf '%s\n' "${!BY_MODEL[@]}" | sort); do
    c=${BY_MODEL[$model]}
    if [[ "$c" -ne "$EXPECTED_PER_MODEL" ]]; then
      echo "  $c	$model  <-- expected $EXPECTED_PER_MODEL"
    else
      echo "  $c	$model"
    fi
  done
}

# -----------------------------------------------------------------------------
# Dispatch by folder name
# -----------------------------------------------------------------------------
case "$BASENAME" in
  benchmarks|benchmarking)
    run_benchmarks
    ;;
  factuality)
    run_factuality
    ;;
  similarities)
    run_similarities
    ;;
  summaries)
    run_summaries
    ;;
  valid_responses)
    run_valid_responses
    ;;
  *)
    echo "Unknown folder type: '$BASENAME'. Supported: benchmarks, benchmarking, factuality, similarities, summaries, valid_responses." >&2
    exit 1
    ;;
esac
