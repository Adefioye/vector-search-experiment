#!/usr/bin/env python3
import csv
from datasets import load_dataset

def main():
    # 1. Download MS MARCO corpus (passages)
    docs_ds = load_dataset("irds/beir_msmarco", "docs", split="docs", trust_remote_code=True)
    docs = {rec["doc_id"]: rec["text"] for rec in docs_ds}

    # 2. Download train queries and relevance judgments (qrels)
    queries_ds = load_dataset("irds/beir_msmarco_train", "queries", split="queries", trust_remote_code=True)
    qrels_ds = load_dataset("irds/beir_msmarco_train", "qrels", split="qrels", trust_remote_code=True)

    # 3. Build a mapping from query_id -> list of relevant passage_ids
    qrels_map = {}
    for rec in qrels_ds:
        qid = rec["query_id"]
        if rec["relevance"] > 0:
            qrels_map.setdefault(qid, []).append(rec["doc_id"])

    # 4. Collect up to 10 unique queries with one relevant passage each
    pairs = []
    for qrec in queries_ds:
        qid, qtext = qrec["query_id"], qrec["text"]
        if qid in qrels_map:
            # pick the first available relevant passage
            for pid in qrels_map[qid]:
                if pid in docs:
                    pairs.append((qtext, docs[pid]))
                    break
        if len(pairs) >= 10:
            break

    # 5. Write out to TSV: query \t relevant_passage
    output_file = "msmarco_examples.tsv"
    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        for query_text, passage_text in pairs:
            writer.writerow([query_text, passage_text])

    print(f"✅ Wrote {len(pairs)} unique query–passage pairs to '{output_file}'")

if __name__ == "__main__":
    main()
