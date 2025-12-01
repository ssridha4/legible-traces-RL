#!/bin/bash

python src/utils/create_confidence_dpo_dataset.py \
    --dataset_name "gsm8k" \
    --llm_name "Qwen/Qwen3-8B" \
    --slm_name "Qwen/Qwen3-0.6B" \
    --trace_len 0.75 \
    --slm_prompt_variant "completions_no_reasoning" \
    --base_dir "./outputs" \
    --dataset_split "test" 