

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