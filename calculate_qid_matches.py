#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# ---------------------------
# Config (arrays as requested)
# ---------------------------
BASE_DIR = "jsonl_before_reranking"
OUTPUT_DIR = "before_reranking_qid_match_reports"
TOPK = 20

beir_datasets = [
    # "msmarco", "fiqa", "scifact",
    # "nfcorpus", "arguana", "scidocs"
    "trec-covid", "webis-touche2020", "climate-fever",
]

query_types = ["keywords", "titles", "claims", "questions", "random", "msmarco"]

retrievers = [
    "nomic-embed-text-v1",
    "modernbert-embed-base",
]

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def analyze_file(path, topk=TOPK):
    """
    Analyze one JSONL file at `path`.

    Returns:
      dict(total=int, first_match=int, any_match=int, missing=bool)
    """
    if not os.path.isfile(path):
        return {"total": 0, "first_match": 0, "any_match": 0, "missing": True}

    total = 0
    first_match = 0
    any_match = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(path)}", leave=False):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Make sure we handle missing keys defensively and compare as strings
            qid = str(obj.get("qid", ""))
            passages = obj.get("passages", [])[:topk]
            total += 1
            if not passages:
                continue

            def get_docid(p):
                return str(p.get("docid", ""))

            # qid == first passage docid?
            if qid and qid == get_docid(passages[0]):
                first_match += 1

            # qid appears anywhere in topk?
            if qid and any(qid == get_docid(p) for p in passages):
                any_match += 1

    return {"total": total, "first_match": first_match, "any_match": any_match, "missing": False}


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def main():
    # Iterate over retrievers and datasets; write one report file per (retriever, dataset)
    for retriever in retrievers:
        for dataset in beir_datasets:
            # Accumulate per query_type + overall summary
            per_type_stats = {}
            sum_total = 0
            sum_first = 0
            sum_any = 0
            missing_files = 0

            for qtype in query_types:
                filename = f"{retriever}_{dataset}-queries-{qtype}.jsonl"
                path = os.path.join(BASE_DIR, filename)

                stats = analyze_file(path, topk=TOPK)
                per_type_stats[qtype] = stats

                if stats.get("missing", False):
                    missing_files += 1
                else:
                    sum_total += stats["total"]
                    sum_first += stats["first_match"]
                    sum_any += stats["any_match"]

            # Build report text
            lines = []
            lines.append("=" * 60)
            lines.append(f"Retriever: {retriever}")
            lines.append(f"Dataset  : {dataset}")
            lines.append(f"Base dir : {BASE_DIR}")
            lines.append(f"Top-k    : {TOPK}")
            lines.append("=" * 60)
            lines.append("Per-query-type results:")

            for qtype in query_types:
                st = per_type_stats[qtype]
                if st.get("missing", False):
                    lines.append(f"- {qtype:12s}: MISSING (file not found)")
                else:
                    T, F, A = st["total"], st["first_match"], st["any_match"]
                    lines.append(
                        f"- {qtype:12s}: total={T:6d} | "
                        f"qid==rank1={F:6d} ({pct(F,T):6.2f}%) | "
                        f"qid∈top{TOPK}={A:6d} ({pct(A,T):6.2f}%)"
                    )

            lines.append("")
            lines.append(f"Summary over {len(query_types) - missing_files} found file(s):")
            lines.append(
                f"  total={sum_total:6d} | "
                f"qid==rank1={sum_first:6d} ({pct(sum_first, sum_total):6.2f}%) | "
                f"qid∈top{TOPK}={sum_any:6d} ({pct(sum_any, sum_total):6.2f}%) | "
                f"missing_files={missing_files}"
            )
            lines.append("=" * 60)
            report_text = "\n".join(lines)

            # Write one file per (retriever, dataset)
            out_name = f"{retriever}_{dataset}_qid_match_report.txt"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(report_text)

            # Also echo brief status
            print(f"[✔] Wrote report: {out_path}")


if __name__ == "__main__":
    main()
