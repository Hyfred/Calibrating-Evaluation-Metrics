#!/bin/bash

datasets=("Bigbench" "MathQA" "MMLU" "OpenBookQA" "SciQ" "TriviaQA" "TruthfulQA")

for dataset in "${datasets[@]}"; do
  echo $dataset
  python -m src.data_acquisition.prepare_raw.$(echo "$dataset" | awk '{print tolower($0)}')
done