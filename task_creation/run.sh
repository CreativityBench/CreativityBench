#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Export your API key before running this script."
  exit 1
fi

mkdir -p outputs

python 1_tight_clustering.py
python 2_sample_compare.py
python 3_task_creation.py
