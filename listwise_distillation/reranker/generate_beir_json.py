import argparse
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the BEIR dataset to download (e.g., msmarco, scifact, nfcorpus)')
    args = parser.parse_args()
    dataset = args.dataset

    print(f"[📥] Loading dataset BeIR/{dataset} ...")
    ds = load_dataset(f"BeIR/{dataset}", "corpus")

    output_dir = f"beir/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Save queries as JSON
    queries = ds["queries"]
    query_dict = {q["_id"]: q["text"] for q in queries}
    query_path = os.path.join(output_dir, "queries.json")
    with open(query_path, "w") as f:
        json.dump(query_dict, f, indent=2)
    print(f"[✅] Saved queries to {query_path}")

    # Save corpus as JSONL
    corpus = ds["corpus"]
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_path, "w") as f:
        for item in tqdm(corpus, desc="Writing corpus"):
            docid = item["_id"]
            text = item["text"]
            f.write(json.dumps({"_id": docid, "text": text}) + "\n")
    print(f"[✅] Saved corpus to {corpus_path}")

if __name__ == "__main__":
    main()
