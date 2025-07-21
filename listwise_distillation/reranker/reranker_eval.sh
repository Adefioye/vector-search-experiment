#!/bin/bash

# List of BEIR datasets
datasets=(
  "trec-covid"
  "nfcorpus"
  "fiqa"
  "scidocs"
  "arguana"
  "webis-touche2020"
  "climate-fever"
  "scifact"
)

# Specify your model name (change this as appropriate)
model_name="bge-base-en-v1.5"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
  echo "============================================"
  echo "Evaluating dataset: $dataset"
  echo "============================================"

  run_file="results/run.rankt5.${model_name}.${dataset}.txt"

  if [[ ! -f "$run_file" ]]; then
    echo "[⚠️] Run file $run_file not found. Skipping..."
    continue
  fi

  # nDCG@10
  echo "[📊] nDCG@10:"
  python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    "$run_file"

  echo

  # Recall@100
  echo "[📊] Recall@100:"
  python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    "$run_file"

  echo
done
