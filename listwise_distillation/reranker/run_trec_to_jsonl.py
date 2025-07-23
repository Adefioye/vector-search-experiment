import os
import sys
from rerank_utils import trec_to_jsonl

# Add reranker/ directory to PYTHONPATH
# On cloud GPU, use /workspace/vector-search-experiment/listwise_distillation
base_path = os.path.expanduser("/workspace/vector-search-experiment/listwise_distillation")
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
topk = 100

for dataset in datasets:
    print(f"\n[🚀] Running trec_to_jsonl for dataset: {dataset}")

    run_path = f"results/run.beir.{model_name}.{dataset}.txt"
    query_path = f"beir_datasets/{dataset}/queries.json"
    corpus_path = f"beir_datasets/{dataset}/corpus.jsonl"
    output_jsonl = f"rerank_inputs/{dataset}.jsonl"

    if not (os.path.exists(run_path) and os.path.exists(query_path) and os.path.exists(corpus_path)):
        print(f"[⚠️] Missing files for {dataset}, skipping...")
        continue

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    trec_to_jsonl(
        run_path=run_path,
        query_path=query_path,
        corpus_path=corpus_path,
        output_jsonl=output_jsonl,
        topk=topk
    )

    print(f"[✅] Saved {output_jsonl}")
