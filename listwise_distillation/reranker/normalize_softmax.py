#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def softmax(x, T=1.0):
    """Numerically stable softmax with temperature."""
    x = np.asarray(x, dtype=np.float64) / max(T, 1e-12)
    x = x - x.max()  # stability
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        # fallback to uniform if something went wrong
        return np.ones_like(ex) / len(ex)
    return ex / s

def normalize_file(
    in_path,
    out_path,
    passages_key="passages",
    score_key="score",
    temperature=1.0,
    keep_topk=20,          # e.g., 20 to keep top-20 by prob
    sort_by_prob=True        # re-sort passages by prob desc before truncation
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Normalizing (softmax)"):
            obj = json.loads(line)
            passages = obj.get(passages_key, [])
            if not passages:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # collect raw scores
            raw = []
            for p in passages:
                s = p.get(score_key, 0.0)
                try:
                    s = float(s)
                except Exception:
                    s = 0.0
                raw.append(s)

            probs = softmax(raw, T=temperature)

            # overwrite score field with probability
            for p, pr in zip(passages, probs):
                p[score_key] = float(pr)

            # optionally re-sort by prob and keep top-k
            if sort_by_prob:
                passages.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)

            if keep_topk is not None and keep_topk > 0:
                passages = passages[:keep_topk]

            obj[passages_key] = passages
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-query softmax normalization of passage scores (overwrite original score).")
    ap.add_argument("--input", required=True, help="Path to input JSONL.")
    ap.add_argument("--output", required=True, help="Path to output JSONL.")
    ap.add_argument("--passages-key", default="passages", help="Key for passages list.")
    ap.add_argument("--score-key", default="score", help="Key of score in each passage.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (lower=sharper).")
    ap.add_argument("--keep-topk", type=int, default=20, help="Keep top-K passages by probability (0=keep all).")
    ap.add_argument("--no-sort", action="store_true", help="Do not re-sort by probability.")
    args = ap.parse_args()

    # Normalize the scores in the input file and save to output file for each retriever and dataset and query type
    print(f"Normalizing scores in {args.input} to {args.output} with temperature {args.temperature}, "
          f"keeping top-{args.keep_topk} passages, sort by prob: {not args.no_sort}")
    if args.keep_topk < 0:
        raise ValueError("keep_topk must be non-negative, use 0 to keep all passages.")

    beir_datasets = ['scifact', 'nfcorpus']

    query_types = ['titles', 'claims', 'questions', 'random', 'msmarco', 'keywords']

    retrievers = ['nomic-embed-text-v1', 'modernbert-embed-base']

    for retriever in retrievers:
        for beir_dataset in beir_datasets:
            for query_type in query_types:
                input_file = f'outputs/{retriever}_{beir_dataset}-queries-{query_type}.jsonl'
                output_file = f'final_outputs/{retriever}_{beir_dataset}-queries-{query_type}-normalized.jsonl'
                print(f"Processing {input_file} -> {output_file}")
                normalize_file(
                    in_path=input_file,
                    out_path=output_file,
                    passages_key=args.passages_key,
                    score_key=args.score_key,
                    temperature=args.temperature,
                    keep_topk=args.keep_topk,
                    sort_by_prob=(not args.no_sort),
                )
                # Remove the old file after normalization
                os.remove(input_file)
                print(f"Removed old file: {input_file}")
