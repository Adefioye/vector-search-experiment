#!/bin/bash

datasets=(
  "scifact"
#   "arguana"
#   "nfcorpus"
#   "scidocs"
#   "fiqa"
#   "trec-covid"
#   "webis-touche2020"
#   "quora"
)

for dataset in "${datasets[@]}"
do
  echo "Running evaluation for $dataset"
  python evaluate_beir.py --dataset_name "$dataset"
done
