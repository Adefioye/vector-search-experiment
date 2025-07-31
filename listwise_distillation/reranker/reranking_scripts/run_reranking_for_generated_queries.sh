#!/usr/bin/env bash
# ------------------------------------------------------------
# Rerank every (retriever, dataset, query_type) triple
# ------------------------------------------------------------
set -euo pipefail

# ----------------- 1. Parameter lists -----------------------
beir_datasets=(
  msmarco fiqa scifact trec-covid nfcorpus
  arguana webis-touche2020 scidocs climate-fever
)

query_types=(
  keywords titles claims questions random msmarco
)

retrievers=(
  nomic-embed-text-v1
  nomic-embed-text-v1-unsupervised
  modernbert-embed-base
  modernbert-embed-base-unsupervised
)

# -------------- 2. Reranker model to apply ------------------
model_name="Soyoung97/RankT5-3b"
# model_name="BAAI/bge-reranker-v2.5-gemma2-lightweight"  # ← swap if you like

# -------------- 3. Loop over all combinations ---------------
for retriever in "${retrievers[@]}"; do
  for dataset in "${beir_datasets[@]}"; do
    for qtype in "${query_types[@]}"; do

      echo "====================================================="
      echo "🔄  Reranker: $model_name"
      echo "🔵  Retriever: $retriever"
      echo "📚  Dataset:   $dataset"
      echo "🔑  QueryType: $qtype"
      echo "====================================================="

      input_path="jsonl_before_reranking/${retriever}_${beir_dataset}-queries-${qtype}.jsonl"
      output_path="jsonl_after_reranking/${retriever}_${beir_dataset}-queries-${qtype}.jsonl"

      if [[ ! -f "$input_path" ]]; then
        echo "[⚠️ ] Input file not found: $input_path  – skipping."
        continue
      fi

      python listwise_distillation/reranker/reranking.py \
        --model_name  "$model_name" \
        --input_path  "$input_path" \
        --output_path "$output_path"

      echo "[✅] Finished → $output_path"
      echo
    done
  done
  echo "Finished reranking for retriever: $retriever"
  echo "====================================================="
  echo
done
