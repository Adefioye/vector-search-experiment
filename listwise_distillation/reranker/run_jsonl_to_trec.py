import os
import sys
from rerank_utils import jsonl_to_trec

# Add reranker/ directory to PYTHONPATH
# On cloud GPU, use /workspace/vector-search-experiment/listwise_distillation
base_path = os.path.expanduser("/workspace/vector-search-experiment/listwise_distillation/reranker")
sys.path.append(base_path)

# === Configuration ===
datasets = [
    "nfcorpus",
    "scifact",
    # "trec-covid",
    # "fiqa",
    # "scidocs",
    # "arguana",
    # "webis-touche2020",
    # "climate-fever"
]

# BAAI/bge-base-en-v1.5
model_name = "nomic-embed-text-v1"  # or any other model you're evaluating

# === Conversion Loop ===
for dataset in datasets:
    print(f"[🔁] Converting {dataset} rerank JSONL to TREC...")

    input_jsonl = f"rerank_outputs/{dataset}.rankt5.jsonl"
    output_trec = f"../encoding/results/run.rankt5.{model_name}.{dataset}.txt"

    if not os.path.exists(input_jsonl):
        print(f"[⚠️] Missing {input_jsonl}, skipping...")
        continue

    os.makedirs(os.path.dirname(output_trec), exist_ok=True)
    jsonl_to_trec(input_jsonl, output_trec, tag="rerank")


