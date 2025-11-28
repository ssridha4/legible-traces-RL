#!/bin/bash

# Set variables
PROMPT_VARIANTS=("default" "numbered" "self_check" "structured")
CONFIG="configs/gsm8k/test/600M/no_reasoning_completion.yaml"
TRACE_LEN=0.75
LIMIT=""    # Empty means "not provided"
SAVE_LOGPROBS=False

for VARIANT in "${PROMPT_VARIANTS[@]}"; do
    echo "Running for prompt variant: $VARIANT"

    TRACE_CSV="./outputs/extracted_traces/gsm8k/Qwen/Qwen3-8B/test/extracted_traces_${VARIANT}.csv"

    CMD=(
        python src/generate_completions.py
        --config "$CONFIG"
        --trace_csv_file "$TRACE_CSV"
        --trace_len "$TRACE_LEN"
        --prompt_variant "completions_no_reasoning"
        --save_logprobs "$SAVE_LOGPROBS"
    )

    # Add limit only if non-empty
    if [[ -n "$LIMIT" ]]; then
        CMD+=(--limit "$LIMIT")
    fi

    "${CMD[@]}"
done
