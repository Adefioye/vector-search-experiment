

# Set up
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
wandb login
huggingface-cli login
```

> Use `tmux` to manage code execution to avoid program shutting down when screen is off

## TODO
### Next steps
- [ ] Perform knowledge distillation of `ModernBERT` (Ongoing)
- [ ] Evaluate the new model on BEIR & TREL datasets & MS Marco Dev (maybe!)