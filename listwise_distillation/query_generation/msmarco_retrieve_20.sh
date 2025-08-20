#!/usr/bin/env bash
# ------------------------------------------------------------
# Encode + retrieve on MS MARCO for all four embed models
# ------------------------------------------------------------
set -euo pipefail

# ---------------- 1. Model triplets -------------------------
models=(
  # "nomic-ai/nomic-embed-text-v1"
  # "nomic-ai/nomic-embed-text-v1-unsupervised"
  "nomic-ai/modernbert-embed-base"
  "nomic-ai/modernbert-embed-base-unsupervised"
)

model_names=(
  # "nomic-embed-text-v1"
  # "nomic-embed-text-v1-unsupervised"
  "modernbert-embed-base"
  "modernbert-embed-base-unsupervised"
)

model_prefixes=(
  # "nomic-ai"
  # "nomic-ai"
  "nomic-ai"
  "nomic-ai"
)

query_types=(keywords titles claims questions random msmarco)
dataset="msmarco"

# --------------- 2. Helper functions ------------------------
encode() {   # $1=model
  python listwise_distillation/encoding/encode_corpus.py \
        --model_name "$1" --normalize --pooling mean \
        --batch_size 1800 --dataset "$dataset"

    echo "✅  Encoding completed for model: $1 on dataset: $dataset"
}

search() {   # $1=model  $2=pfx  $3=name
  for qt in "${query_types[@]}"; do
    python -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder "$1" --l2-norm \
      --query-prefix "search_query: " \
      --index "indices/${2}_${3}_${dataset}_index" \
      --topics "generated_queries/${dataset}_generated_queries_${qt}.tsv" \
      --output "retrieval_runs/run.${3}.${dataset}.generated-queries-${qt}_20.txt" \
      --hits 20 --device cuda:0
    echo "✅  $3 | $dataset | $qt"
  done
}

# --------------- 3. Main loop -------------------------------
for i in "${!models[@]}"; do
  model="${models[$i]}"
  mname="${model_names[$i]}"
  pfx="${model_prefixes[$i]}"

  echo "================================================="
  echo "🔵 Model: $model   (dataset: $dataset)"
  echo "================================================="

  encode "$model"
  search "$model" "$pfx" "$mname"
  # keep the msmarco index for possible reuse (no rm -rf)
done
