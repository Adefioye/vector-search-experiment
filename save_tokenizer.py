from transformers import AutoTokenizer
dataset = 'fiqa'
# model_name = 'modernbert-embed-base'
model_name = 'nomic-embed-text-v1'
model=f'models/{model_name}_{dataset}-infonce-loss'
pretrained_model_name = "nomic-ai/nomic-embed-text-v1"  # Use the base model for tokenizer
finetuned_model_name = model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
tokenizer.save_pretrained(finetuned_model_name)
