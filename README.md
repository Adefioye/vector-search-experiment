

# Set up
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
# [linux only] cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# [linux & win] cuda 12.6 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
# [linux & win] cuda 12.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
# [linux only] (EXPERIMENTAL) rocm 6.3 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm6.3
MAX_JOBS=4 pip install flash-attn --no-build-isolation
wandb login
huggingface-cli login
```
- Use the below to avoid mouse pasting ascii letters in the terninal
```
export TERM=xterm-256color
```
> Use `tmux` to manage code execution to avoid program shutting down when screen is off

## TODO
### Next steps
- [ ] Finetune using CMNRL, MNRL with batch size of 512 `ModernBERT-base` (Ongoing)
- [ ] Evaluate the new model on BEIR & TREL datasets & MS Marco Dev (maybe!)


```
python cadet-dense-retrieval/encoding/encode_beir_corpus.py --model_name answerdotai/ModernBERT-base --normalize --pooling cls --batch_size 1800 --dataset scifact

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder models/answerdotai/ModernBERT-base --l2-norm --query-prefix "Represent this sentence for searching relevant passages: " \
    --index indices/models_ModernBERT-base_scifact_index \
    --topics beir-v1.0.0-scifact-test \
    --output run.beir.ModernBERT-base.scifact.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-scifact-test \
    run.beir.ModernBERT-base.scifact.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-scifact-test \
    run.beir.ModernBERT-base.scifact.txt

    rm -r indices/models_ModernBERT-base_scifact_index

```