#!/usr/bin/env bash
# Reproduction script for Submission 18 (Leaderboard ID: 1007061)
# Uses official Codabench GPU runtime: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

set -euo pipefail

INPUT_DATA_DIR="${1:-./mock_input_data}"
OUTPUT_DIR="${2:-./output_sub18}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMISSION_DIR="$ROOT_DIR/inference/submission_18"
INGESTION_SCRIPT="$SCRIPT_DIR/ingestion.py"

mkdir -p "$OUTPUT_DIR"

echo "=== Running Submission 18 (ViTPose-Large Ensemble 256+384, 4-scale TTA, GPU DARK) ==="
echo "Input: $INPUT_DATA_DIR"
echo "Output: $OUTPUT_DIR"

docker run --rm --gpus all --network none \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$SUBMISSION_DIR":/app/submission \
    -v "$INPUT_DATA_DIR":/app/input_data:ro \
    -v "$INGESTION_SCRIPT":/app/ingestion/ingestion.py:ro \
    -v "$OUTPUT_DIR":/app/output \
    -w /app/submission \
    pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
    python /app/ingestion/ingestion.py \
        --input /app/input_data \
        --submission /app/submission \
        --output /app/output/predictions.json \
        --time-limit 7200

echo "=== Finished! Predictions saved to $OUTPUT_DIR/predictions.json ==="

