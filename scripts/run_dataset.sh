#!/usr/bin/env bash
set -euo pipefail

# Batch-predict thermal conductivity from a CSV with a 'composition' column
# (no S3 required).
# Usage: ./scripts/run_dataset.sh [csv_file]
#
# Example:
#   ./scripts/run_dataset.sh data/input.csv
# Outputs are written to ./data/output/ (predictions.csv, predictions_hist.png)

CSV_FILE="${1:-data/input.csv}"

uv run model-run --local \
  --parameters "$(jq -n \
    --arg file "file://${CSV_FILE}" \
    '{mode:"dataset", inputs:{file:{uri:$file, mime_type:"text/csv"}}, parameters:{}}')"
