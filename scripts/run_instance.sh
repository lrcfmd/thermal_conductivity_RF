#!/usr/bin/env bash
set -euo pipefail

# Predict the thermal conductivity of one composition (no S3 required).
# Usage: ./scripts/run_instance.sh [composition]
#
# Example:
#   ./scripts/run_instance.sh Bi2Te3

COMPOSITION="${1:-Bi2Te3}"

uv run model-run --local \
  --parameters "$(jq -n \
    --arg comp "${COMPOSITION}" \
    '{mode:"instance", inputs:{composition:{value:$comp}}, parameters:{}}')"
