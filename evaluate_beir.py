"""Evaluation script for BEIR datasets with a PLAID index."""

from __future__ import annotations

import argparse
import os
import json


from pylate import evaluation, indexes, models, retrieve
import wandb
import pandas as pd


if __name__ == "__main__":
    query_len = {
        "quora": 32,
        "climate-fever": 64,
        "nq": 32,
        "msmarco": 32,
        "hotpotqa": 32,
        "nfcorpus": 32,
        "scifact": 48,
        "trec-covid": 48,
        "fiqa": 32,
        "arguana": 64,
        "scidocs": 48,
        "dbpedia-entity": 32,
        "webis-touche2020": 32,
        "fever": 32,
        "cqadupstack/android": 32,
        "cqadupstack/english": 32,
        "cqadupstack/gaming": 32,
        "cqadupstack/gis": 32,
        "cqadupstack/mathematica": 32,
        "cqadupstack/physics": 32,
        "cqadupstack/programmers": 32,
        "cqadupstack/stats": 32,
        "cqadupstack/tex": 32,
        "cqadupstack/unix": 32,
        "cqadupstack/webmasters": 32,
        "cqadupstack/wordpress": 32,
    }

    # Parse dataset_name from command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="scifact",
        help="Name of the dataset to evaluate on (default: 'scifact')",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = "answerdotai/ModernBERT-base"

    OUTPUT_DIR = "outputs/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{model_name.split('/')[-1]}/{dataset_name}_scores.json")

    ## Initialize Weights & Biases (wandb) for logging
    wandb.init(
        project="modernbert-beir-eval",  # change to whatever project name you like
        name=f"{model_name.split('/')[-1]}-{dataset_name}",
        config={
            "dataset": dataset_name,
            "model": model_name,
        },
        tags=["beir", dataset_name],
        group=model_name.split("/")[-1],
    )

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    index = indexes.PLAID(
        override=True,
        index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
    )

    retriever = retrieve.ColBERT(index=index)

    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=2000,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                # Remove the query_id from the score
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        # queries=queries,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(evaluation_scores, f, indent=2)

    print(f"Evaluation scores saved to {OUTPUT_PATH}")

    # Log to wandb Table
    scores_table = pd.DataFrame([evaluation_scores])
    wandb.log({"eval_scores": wandb.Table(dataframe=scores_table)})

    wandb.finish()