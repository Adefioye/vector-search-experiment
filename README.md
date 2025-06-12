

# Set up
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
wandb login
huggingface-cli login
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
- Use the below to avoid mouse pasting ascii letters in the terninal
```
export TERM=xterm-256color
```
> Use `tmux` to manage code execution to avoid program shutting down when screen is off

## TODO
### Next steps
- [ ] Finetune using CMNRL(CachedMultipleNegativesRankingLoss) with batch size of 512 `ModernBERT-base` (Ongoing)
- [ ] Evaluate the new model on BEIR & TREL datasets & MS Marco Dev (maybe!)