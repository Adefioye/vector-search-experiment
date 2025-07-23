python - <<'PY'
from datasets import load_dataset, DatasetDict, concatenate_datasets
import json, os, tqdm, gzip, pathlib

dset = "scifact"
out_dir = f"beir/{dset}"
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset(f"BeIR/{dset}", "corpus")

# save queries.json
queries = {q["_id"]: q["text"] for q in ds["queries"]}
with open(f"{out_dir}/queries.json", "w") as f:
    json.dump(queries, f)

# save corpus.jsonl
with open(f"{out_dir}/corpus.jsonl", "w") as f:
    for row in ds["corpus"]:
        f.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")
print("[✔] Scifact corpus & queries saved.")
PY
