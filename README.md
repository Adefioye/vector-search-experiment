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
- [x] Run evals on `nomic-embed-text-v1-unsupervised`, `modernbert-embed-base-unsupervised` and `modernbert-embed-base` using `NDCG@10` and `Recall@100`
- [ ] Create synthetic queries 
- [ ] Finetune on specific __Scifact__,  __FiQA__, __TREC-Covid__, __NFCorpus__, and __MSMARCO__ datasets using contrastive and listwise distillation loss and evaluate performance using `NDCG@10` and `Recall@100`
- [ ] If combined loss(contrastive + listwise distillation) is better, Finetune on all `8 datasets` we did eval on retrieval and reranking on.
- [ ] Finetune supervised and unsupervised model on the different query types and evaluate on `DL19, DL20, Scifact, FiQA, TREC-Covid, NFCorpus and MSMARCO` using `NDCG@10`, __Table 4__.
- [ ] Finetune supervised and unsupervised model on `human written` vs `synthetic` queries and evaluate on `DL19, DL20, Scifact, FiQA, TREC-Covid, NFCorpus and MSMARCO` using `NDCG@10`, __Table 5__.


### Steps

### Running evals with or without rerankers
1. Run evals on base retriever and reranker. Script `run_eval_with_reranker.sh` executes all steps below:
  - Run `test_bge_beir.sh`, produces `results/run.beir.${model_name}.${dataset}.txt` (NOTE: `Add model_name in this file`)
  - Create `json/jsonl` for queries and corpus file using `run_generate_beir_json.sh`
  - Convert  `initial trec result` to `jsonl` by running `run_trec_to_jsonl.py` (NOTE: `Add model_name in this file`)
  - Run reranker using `run_reranking.sh`
  - Convert reranked `jsonl` to `trec` using `run_jsonl_to_trec.py` (NOTE: `Add model_name in this file`)
  - Run eval using pyserini using `rerank_eval.sh` (NOTE: `Add model_name in this file`)
2. Run evals for single retrieval purposes. Script `run_eval_without_reranker.sh`.

### Filtering synthetic queries (IDEATING)
1. Generate synthetic queries for all datasets by running all scripts in `query_generation/query_gen_scripts` folder.
2. Generates top100 hits using `bash bge_retrieve.sh`
3. Generate ignore list using `corpora_deduplication.py` and `msmarco_get_train_ids.py`[DONE I SUPPOSE]
4. Filter each run to produce synthetic queries whose passage ranks first in the top20 using `retriever_filtering_step.py`
5. Generate reranker input jsonl using `generate_jsonl_for_reranking.py` (Might not need this for finetuning. Perhaps only applicable to `general retrieval model in the paper`)

>NOTE:
1. Might need to add `rank > 20 break`
```
if (pid not in disregard_ids):
    vals[3] = str(rank)
    rank += 1
    step_one_filtered_lines.append(' '.join(vals).strip())
```
2. Create `msmarco_examples.tsv` needed for few-shot msmarco synthetic queries.

### Finetuning
1. For `infonce loss`, use `train_nomic_embed_infonce_loss.py`.
2. For `infonce + listwise loss`, use `train_nomic_embed_joint_loss.py`.

# TODO
### To generate synthetic queries
```
chmod +x listwise_distillation/query_generation/query_gen_scripts/run_all_query_generation.sh
./listwise_distillation/query_generation/query_gen_scripts/run_all_query_generation.sh
```

### How to preprocess data for finetuning using synthetic queries (My understanding)
1. Generate `top 20` hits using `model_retrieve_20.sh`.
2. Filter out the queries whose positive passage is not in `top 20` rank using `retriever_filtering_step.py`.
3. Ideally use crossencoder `reranker` RankT5 to re-score the scores in `stage 2` and create a `jsonl`.
  - Run trec to jsonl `per retriever per dataset per query_type` using `generate_jsonl_for_reranking.py`.
4. Run reranking `per retriever per dataset per query_type` using `run_reranking_for_generated_queries.sh`
5. Filter out the queries whose positive passage is in top 20 rank using `filter_by_reranker.py`
6. The cross-encoder score is normalized using `normalize_scores.py`
7. The normalized outputs is then passed to `create_train_dev_data.py`
8. Use the jsonl file to perform both `infonce and listwise distillation training`

- __INPUTS & OUTPUTS & How to achieve the steps above?__
1. 
  - input: `generated_queries/${dataset}_generated_queries_${query_type}.tsv` (qid, text).
  - output: `retrieval_runs/run.${model_name}.${dataset}.generated-queries-${query_type}_20.txt`.
2. 
  - input: `retrieval_runs/run.${model_name}.${dataset}.generated-queries-${query_type}_20.txt` (TREC format)
  - output: `retrieval_runs/run.{retriever}.{beir_dataset}.generated-queries-{query_type}.filtered.txt` (TREC format)
3. To get to achieve 3, we need to do the following:
  - Usually, the output coming from `step 2` is in TREC format. Convert to `jsonl`.
  - input: `retrieval_runs/run.{retriever}.{beir_dataset}.generated-queries-{query_type}.filtered.txt`
  - output: `jsonl_before_reranking/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
4. Rerank each `per retriever per dataset per query_type` results using `run_reranking_for_generated_queries.sh`.
  - input: `jsonl_before_reranking/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
  - output: `jsonl_after_reranking/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
5. After this, we can then launch `filter_by_reranker.py`
  - input: `jsonl_after_reranking/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
  - output: `outputs/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
6. Normalize output from reranker using `normalized_scores.py`. Use `score` instead of `rankt5_score` in `json passages`.
  - input: `outputs/{retriever}_{beir_dataset}-queries-{query_type}.jsonl`
  - output: `final_outputs/{retriever}_{beir_dataset}-queries-{query_type}-normalized.jsonl`
7. Create train and dev split 
  - input: for all query types 
        `final_outputs/{retriever}_{beir_dataset}-queries-{query_type}-normalized.jsonl`
  output: 
      `final_data/beir.{self.retriever}.{self.dataset}.train.generated_queries.listwise.jsonl`
      `final_data/beir.{self.retriever}.{self.dataset}.dev.generated_queries.listwise.jsonl`

>NOTE:
- Dig into `filter_by_reranker.py`: it appears to take `jsonl` with `rankt5` scores and ensures only queries with top 20 passages whose `positive passage` has `rank 1` are retained
- Dig into `normalized_scores.py`: This takes output `jsonl` from above and normalize the `rankt5` scores.
- To evaluate in-domain effectiveness, we finetune on the MSMARCO passage dataset. For this task, we sample 200K passages (rather than 100K as used elsewhere) to better take advantage of its scale and generality.

### How to generate Table 4?
>NOTE:
Table 4: Retrieval effectiveness when training the E5-unsupervised model with the different query types considered and all generated queries. Best scores overall are bolded and we underline the best scores when training with a single query type.

1. Use `combined loss` if proven to help recover retrieval effectiveness of model.
2. Use each `query type` separately finetune `nomic-embed-supervised vs unsupervised` model
3. Get NDCG@10 for the `finetuned` model

### How to generate Table 5?
>NOTE:
Table 5: Retrieval effectiveness for the E5-unsupervised model fine-tuned with human-written and synthetic queries. For the synthetic queries, results are provided for both a subset of 56K queries to provide a fair comparison and the full query set.

1. We have 5 different queries here
  - User queries (Human) `56K`
  - User queries, Few-shot (Synthetic) `56K`
  - User queries, Few-shot (Synthetic) `96K`
  - User queries, Zero-shot (Synthetic) `56K`
  - User queries, Zero-shot (Synthetic) `96K`


# Training TODOs:
- Train `modernbert-embed-base` and `nomic-embed-text-v1` on scifact and nfcorpus for INFONCE LOSS.
- Evaluate trained model on their respective datasets and obtain NDCG@10 and RECALL@100.
```
chmod run_eval_without_reranker.sh
./run_eval_without_reranker.sh
```
