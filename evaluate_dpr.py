# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import mteb
from sentence_transformers import SentenceTransformer

model_name = "kokolamba/ModernBERT-base-DPR-8e-05-CMNRL-minibs16"
lr = 8e-5
model_shortname = model_name.split("/")[-1]
# run_name = f"{model_shortname}-DPR-{lr}"
run_name = f"{model_shortname}"
output_dir = f"outputs/{model_shortname}/{run_name}/"
model = SentenceTransformer(model_name, device="mps")

task_names = ["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"]
tasks = mteb.get_tasks(tasks=task_names)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{run_name}")