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

python 1_partonomy_graph.py
python 2_physical_attributes.py
python 3_physical_combination.py
python 4_state_attributes.py
python 5_state_combination.py
python 6_functional_affordance.py
python 7_entity_assembly.py
