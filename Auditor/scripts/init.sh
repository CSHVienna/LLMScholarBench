#!/usr/bin/env bash
set -e

# Initialize conda so "conda activate" works when script is run as ./script.sh
CONDA_BASE="${CONDA_EXE%/bin/conda}"  # if conda is in PATH
[[ -z "$CONDA_BASE" ]] && CONDA_BASE="$(conda info --base 2>/dev/null)" || true
for d in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [[ -d "$d" && -f "${d}/etc/profile.d/conda.sh" ]]; then
    source "${d}/etc/profile.d/conda.sh"
    break
  fi
done

conda activate py311_llmscholar
export PYTHONPATH="${PYTHONPATH}:../code/"