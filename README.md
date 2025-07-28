# Set up environment for training
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
wandb login
huggingface-cli login
```
- Use the below to avoid mouse pasting ascii letters in the terninal
```
export TERM=xterm-256color
```
> Use `tmux` to manage code execution to avoid program shutting down when screen is off

# Setup environment for evals
1. When doing evals with pyserini, setup environment with the following:
- Install miniconda and setup eval environment
```
chmod +x install_miniconda.sh
source ~/.bashrc
chmod +x setup_dev_env.sh
bash setup_dev_env.sh
```
2. Make sure to add `trust_remote_code=True` for AutoDocumentEncoder and AutoQueryEncoder

## TODO
### Next steps
- [x] Run evals on `nomic-embed-text-v1` and reranker `RankT5-3b` using `NDCG@10`.
- [ ] Run evals on `nomic-embed-text-v1-unsupervised`, `modernbert-embed-base-unsupervised` and `modernbert-embed-base` using `NDCG@10` and `Recall@100`
- [ ] Create synthetic queries 
- [ ] Finetune on specific __BEIR__ datasets using contrastive and listwise distillation loss and evaluate performance using `NDCG@10` and `Recall@100`


### Steps

## Running evals with or without rerankers
1. Run evals on base retriever and reranker. Script `run_eval_with_reranker.sh` executes all steps below:
  - Run `test_bge_beir.sh`, produces `results/run.beir.${model_name}.${dataset}.txt` (NOTE: `Add model_name in this file`)
  - Create `json/jsonl` for queries and corpus file using `run_generate_beir_json.sh`
  - Convert  `initial trec result` to `jsonl` by running `run_trec_to_jsonl.py` (NOTE: `Add model_name in this file`)
  - Run reranker using `run_reranking.sh`
  - Convert reranked `jsonl` to `trec` using `run_jsonl_to_trec.py` (NOTE: `Add model_name in this file`)
  - Run eval using pyserini using `rerank_eval.sh` (NOTE: `Add model_name in this file`)
2. Run evals for single retrieval purposes. Script `run_eval_without_reranker.sh`.

## Filtering synthetic queries (IDEATING)

1. Generates top100 hits using `bash bge_retrieve.sh`
2. Generate ignore list using `corpora_deduplication.py` and `msmarco_get_train_ids.py`[DONE I SUPPOSE]
3. Filter each run to produce synthetic queries whose passage ranks first in the top20 using `retriever_filtering_step.py`
4. Generate reranker input jsonl using `generate_jsonl_for_reranking.py` (Might not need this for finetuning. Perhaps only applicable to `general retrieval model in the paper`)

>NOTE:
1. Might need to add `rank > 20 break`
```
if (pid not in disregard_ids):
    vals[3] = str(rank)
    rank += 1
    step_one_filtered_lines.append(' '.join(vals).strip())
```
2. Create `msmarco_examples.tsv` needed for few-shot msmarco synthetic queries.
3. 
