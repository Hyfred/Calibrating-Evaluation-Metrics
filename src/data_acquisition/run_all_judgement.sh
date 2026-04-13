#!/bin/bash
#llama3.1
datasets=("TriviaQA" "TruthfulQA")
prompt_names=("ling1s-topk" "verb1s-topk")
qa_llms=(mistral gemma)
judgement_llms=(llama2)

for dataset in "${datasets[@]}"; do
  for prompt_name in "${prompt_names[@]}"; do
    for qa_llm in "${qa_llms[@]}"; do
      for judgement_llm in "${judgement_llms[@]}"; do
        echo $dataset
        echo $prompt_name
        echo $qa_llm
        echo $judgement_llm
        python -m src.data_acquisition.judgement judgement_llm=$judgement_llm qa_llm=$qa_llm prompt_name=$prompt_name dataset_name=$(echo "$dataset" | awk '{print tolower($0)}')
      done
    done
  done
done