from transformers import AutoTokenizer
dataset = 'nfcorpus'
# model_name = 'nomic-embed-text-v1'
model_name = 'modernbert-embed-base'
model=f'models/{model_name}_{dataset}-joint-loss-epoch-2'
pretrained_model_name = "nomic-ai/modernbert-embed-base"  # Use the base model for tokenizer
finetuned_model_name = model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
tokenizer.save_pretrained(finetuned_model_name)
