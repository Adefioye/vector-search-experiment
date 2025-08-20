#!/usr/bin/env bash
# ------------------------------------------------------------
# Encode + retrieve for all (model, dataset, query_type) combos
#   • models  : 4 (nomic & modernBERT, supervised + unsup.)
#   • datasets: 8 (all BEIR collections except MS MARCO)
#   • q-types : 6 (keywords … msmarco)
# Result: retrieval_runs/run.<model_name>.<dataset>.generated-queries-<qt>_20.txt
# ------------------------------------------------------------
set -euo pipefail

# -------------------- 1. Model lists ------------------------
models=(
  # "nomic-ai/nomic-embed-text-v1"
  "nomic-ai/nomic-embed-text-v1-unsupervised"
  "nomic-ai/modernbert-embed-base"
  "nomic-ai/modernbert-embed-base-unsupervised"
)

model_names=(
  # "nomic-embed-text-v1"
  "nomic-embed-text-v1-unsupervised"
  "modernbert-embed-base"
  "modernbert-embed-base-unsupervised"
)

model_prefixes=(
  # "nomic-ai"
  "nomic-ai"
  "nomic-ai"
  "nomic-ai"
)

# -------------------- 2. Dataset / q-type lists --------------
other_datasets=(
  fiqa scifact trec-covid nfcorpus 
  arguana webis-touche2020 scidocs climate-fever
)
query_types=(keywords titles claims questions random msmarco)

# -------------------- 3. Helper functions -------------------
encode() {   # $1=model  $2=dataset
  python listwise_distillation/encoding/encode_corpus.py \
        --model_name "$1" --normalize --pooling mean \
        --batch_size 1800 --dataset "$2"
    echo "✅  Encoding completed for model: $1 on dataset: $2"
}

search() {   # $1=model  $2=prefix  $3=name  $4=dataset
  for qt in "${query_types[@]}"; do
    python -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder "$1" --l2-norm \
      --query-prefix "search_query: " \
      --index "indices/${2}_${3}_${4}_index" \
      --topics "generated_queries/${4}_generated_queries_${qt}.tsv" \
      --output "retrieval_runs/run.${3}.${4}.generated-queries-${qt}_20.txt" \
      --hits 20 --device cuda:0
    echo "✅  $3 | $4 | $qt"
  done
}

# -------------------- 4. Main loop --------------------------
for i in "${!models[@]}"; do
  model="${models[$i]}"
  mname="${model_names[$i]}"
  pfx="${model_prefixes[$i]}"

  echo "================================================="
  echo "🔵 Model: $model"
  echo "================================================="

  for dataset in "${other_datasets[@]}"; do
    echo "---------------------------------------------"
    echo "📚 Dataset: $dataset"
    echo "---------------------------------------------"

    encode "$model" "$dataset"
    search "$model" "$pfx" "$mname" "$dataset"

    # Delete index to free space
    rm -rf "indices/${pfx}_${mname}_${dataset}_index"
  done
done

