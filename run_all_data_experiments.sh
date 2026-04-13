#!/bin/bash

# All datasets
datasets=("mathqa" "mmlu" "openbookqa" "sciq" "triviaqa" "truthfulqa")

# Prompt options
prompts=("ling1s-topk" "verb1s-topk")

# QA LLM options
qa_llms=("mistral" "gemma")

# Where to save logs
log_dir="logs_v4"
mkdir -p "$log_dir"

for dataset in "${datasets[@]}"; do
  for prompt in "${prompts[@]}"; do
    for qa in "${qa_llms[@]}"; do
      echo "Running with dataset=$dataset, prompt=$prompt, qa_llm=$qa"

      log_file="${log_dir}/${dataset}_${prompt}_${qa}.log"

      nohup python experiments.py \
        --dataset_name "$dataset" \
        --prompt_name "$prompt" \
        --qa_llm "$qa" >"$log_file" 2>&1 &
    done
  done
done