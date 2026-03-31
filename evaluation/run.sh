#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-gpt-5.2}"
MODEL_PROVIDER="${MODEL_PROVIDER:-}"

if [[ -z "$MODEL_PROVIDER" ]]; then
  if [[ "$MODEL" == gpt-* || "$MODEL" == o1-* || "$MODEL" == o3-* ]]; then
    MODEL_PROVIDER="openai"
  else
    MODEL_PROVIDER="vllm"
  fi
fi

if [[ "$MODEL_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Export your API key before running evaluation with an OpenAI model."
  exit 1
fi

mkdir -p "${OUTPUT_DIR:-$SCRIPT_DIR/outputs}" "${JUDGED_OUTPUT_DIR:-$SCRIPT_DIR/judged_outputs}"

python evaluate.py

if [[ "${RUN_JUDGE:-1}" == "1" ]]; then
  python judge.py
fi
