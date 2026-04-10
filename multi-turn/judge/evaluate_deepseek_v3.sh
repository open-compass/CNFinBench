#!/bin/bash
# Evaluation script using DeepSeek-V3 model
# Please modify the following variables according to your API configuration

JUDGE_API_KEY="your-api-key"
JUDGE_BASE_URL="https://your-api-url/v1"
JUDGE_MODEL_NAME="deepseek-v3"
OUTPUT_DIR="./output"

# Change to the script's parent directory (multi-turn directory)
cd "$(dirname "$0")/.."

# Run the evaluation script
python -m judge.evaluate \
    --output-dir "$OUTPUT_DIR" \
    --judge-api-key "$JUDGE_API_KEY" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model-name "$JUDGE_MODEL_NAME"
