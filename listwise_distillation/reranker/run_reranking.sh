#!/bin/bash

# List of BEIR datasets
datasets=(
  "nfcorpus"
  "scifact"
#   "trec-covid"
#   "fiqa"
#   "scidocs"
#   "arguana"
#   "webis-touche2020"
#   "climate-fever"
)

# Model to use
model_name="Soyoung97/RankT5-3b"
# model_name="BAAI/bge-reranker-v2.5-gemma2-lightweight"

# Loop over each dataset
for dataset in "${datasets[@]}"
do
  echo "==========================================="
  echo "🚀 Running reranking on: $dataset"
  echo "==========================================="

  input_path="rerank_inputs/${dataset}.jsonl"
  output_path="rerank_outputs/${dataset}.rankt5.jsonl"

  if [[ ! -f "$input_path" ]]; then
    echo "[⚠️] Input file not found: $input_path. Skipping..."
    continue
  fi

  python reranking.py \
    --model_name "$model_name" \
    --input_path "$input_path" \
    --output_path "$output_path" 

  echo "[✅] Finished reranking for $dataset"
  echo
done
