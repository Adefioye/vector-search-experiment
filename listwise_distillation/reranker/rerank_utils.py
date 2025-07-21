import json, gzip, os
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------------
# 1️⃣  TREC → JSONL  (for reranker)
# -------------------------------------------------------------------
def trec_to_jsonl(run_path: str,
                  query_path: str,
                  corpus_path: str,
                  output_jsonl: str,
                  topk: int = 100,
                  is_gzip: bool = False,
                  query_prefix: str = "") -> None:
    """
    Convert a TREC run file + queries + corpus into JSONL:
      { "query": "...", "passages": [ {"docid": "...", "text": "...", "score": ...}, ... ] }

    Args
    ----
    run_path:       TREC run produced by Pyserini (Q0 format)
    query_path:     JSON/JSONL/TSV of queries, keyed by qid
    corpus_path:    JSON/JSONL/TSV of corpus, keyed by docid
    output_jsonl:   path for reranker input
    topk:           keep at most this many hits per query
    is_gzip:        if corpus file is .gz
    query_prefix:   optional prefix (e.g., "search_query: ")
    """
    # ---- 1. read run ----
    trec = defaultdict(list)
    with open(run_path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            trec[qid].append((docid, float(score)))
    # retain top-k
    for qid in trec:
        trec[qid] = sorted(trec[qid], key=lambda x: -x[1])[:topk]

    # ---- 2. read queries ----
    queries = {}
    # supports .json or .tsv (qid\ttext)
    if query_path.endswith(".json"):
        with open(query_path) as f:
            queries = json.load(f)
    else:
        with open(query_path) as f:
            for ln in f:
                qid, text = ln.rstrip("\n").split("\t")
                queries[qid] = text

    # ---- 3. read corpus subset (only needed docids) ----
    wanted = {d for hits in trec.values() for d, _ in hits}
    corpus = {}
    opn = gzip.open if is_gzip else open
    with opn(corpus_path, "rt") as f:
        for ln in f:
            obj = json.loads(ln)
            docid = obj["docid"] if "docid" in obj else obj["_id"]
            if docid in wanted:
                corpus[docid] = obj["text"] if "text" in obj else obj["contents"]

    # ---- 4. write jsonl ----
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as out:
        for qid, hits in trec.items():
            query_text = query_prefix + queries[qid]
            passages = [
                {"docid": docid, "text": corpus[docid], "score": score}
                for docid, score in hits
                if docid in corpus
            ]
            out.write(json.dumps({"qid": qid, "query": query_text,
                                  "passages": passages}) + "\n")

    print(f"[✅] Reranker input saved to {output_jsonl}")


# -------------------------------------------------------------------
# 2️⃣  JSONL (after rerank) → TREC
# -------------------------------------------------------------------
def jsonl_to_trec(rerank_jsonl: str,
                  output_trec: str,
                  tag: str = "rerank") -> None:
    """
    Convert reranked JSONL back to TREC run format.

    The JSONL must contain:
      { "qid": "...", "passages": [ {"docid": "...", "score": ...}, ...] }
    """
    with open(rerank_jsonl) as f, open(output_trec, "w") as out:
        for ln in f:
            obj = json.loads(ln)
            qid = str(obj["qid"])
            for rank, p in enumerate(obj["passages"], start=1):
                out.write(f"{qid} Q0 {p['docid']} {rank} {p['score']} {tag}\n")
    print(f"[✅] TREC file written to {output_trec}")
