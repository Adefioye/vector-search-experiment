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
chmod +x setup_eval_env.sh
bash setup_eval_env.sh
```

## TODO
### Next steps
- [ ] Finetune using CMNRL, MNRL with batch size of 512 `ModernBERT-base` (Ongoing)
- [ ] Evaluate the new model on BEIR & TREL datasets & MS Marco Dev (maybe!)


```
python cadet-dense-retrieval/encoding/encode_beir_corpus.py --model_name kokolamba/ModernBERT-base-DPR-8e-05-CMNRL-minibs128 --normalize --pooling cls --batch_size 1800 --dataset scifact

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder kokolamba/ModernBERT-base-DPR-8e-05-CMNRL-minibs128 --l2-norm --query-prefix "query: "\
    --index indices/kokolamba_ModernBERT-base-DPR-8e-05-CMNRL-minibs128_scifact_index \
    --topics beir-v1.0.0-scifact-test \
    --output results/run.beir.ModernBERT-base-DPR-8e-05-CMNRL-minibs128.scifact.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-scifact-test \
    results/run.beir.ModernBERT-base-DPR-8e-05-CMNRL-minibs128.scifact.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-scifact-test \
    results/run.beir.ModernBERT-base-DPR-8e-05-CMNRL-minibs128.scifact.txt

    rm -r indices/answerdotai_ModernBERT-base_scifact_index

```

2. Try this instead
```
python -m pyserini.encode \
  --input beir-v1.0.0-scifact/corpus.jsonl \
  --output indices/ModernBERT-scifact_index \
  --encoder answerdotai/ModernBERT-base \
  --pooling cls \
  --l2-norm

python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --encoder-class auto \
  --encoder answerdotai/ModernBERT-base \
  --l2-norm \
  --query-prefix "query: " \
  --index indices/ModernBERT-scifact_index \
  --topics beir-v1.0.0-scifact-test \
  --output results/run.ModernBERT.scifact.txt \
  --hits 1000 \
  --remove-query

  # nDCG@10
python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 beir-v1.0.0-scifact-test \
  results/run.ModernBERT.scifact.txt

# Recall@100
python -m pyserini.eval.trec_eval \
  -c -m recall.100 beir-v1.0.0-scifact-test \
  results/run.ModernBERT.scifact.txt

```