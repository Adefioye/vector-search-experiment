# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import mteb
from sentence_transformers import SentenceTransformer
from datetime import datetime

import torch, gc

torch.cuda.empty_cache()
gc.collect()

model_name = "answerdotai/ModernBERT-base"
lr = 8e-5
model_shortname = model_name.split("/")[-1]
# run_name = f"{model_shortname}-DPR-{lr}"
run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{model_shortname}"
output_dir = f"outputs/{model_shortname}/{run_name}/"
model = SentenceTransformer(model_name, device="mps")

# task_names = ["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"]
task_names = ["SciFact"]
tasks = mteb.get_tasks(tasks=task_names)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model,
    output_folder=f"results/{run_name}",
    encode_kwargs={"batch_size": 64}
)