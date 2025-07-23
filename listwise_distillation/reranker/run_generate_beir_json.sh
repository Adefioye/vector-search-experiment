#!/bin/bash

# List of BEIR datasets
datasets=(
  # "nfcorpus"
  "scifact"
  # "trec-covid"
  # "fiqa"
  # "scidocs"
  # "arguana"
  # "webis-touche2020"
  # "climate-fever"
)

# Path to your Python script
SCRIPT_PATH="generate_beir_json.py"

# Loop through each dataset and run the Python script
for dataset in "${datasets[@]}"
do
  echo "Processing dataset: $dataset"
  python $SCRIPT_PATH --dataset $dataset
done
