import json
from tqdm import tqdm
import numpy as np
import random
import os

# Define function to filter based on 'qid' and first 'docid' in 'passages'
def filter_samples(file_path):
    """
    Keeps samples where 'qid' matches the first 'docid' in 'passages', meaning the corresponding passage ranked first for the generated query.
    
    Args:
        file_path (str): Path to the input JSONL file.
        
    Returns:
        list: Filtered list of JSON objects that meet the criteria.
    """
    filtered_samples = []
    total_count = 0

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Skipping.")
        return []

    with open(file_path, 'r', encoding='utf-8') as input_f:
        for line in tqdm(input_f, desc=f'Filtering {file_path}'):
            total_count += 1
            try:
                jsonl_line = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")
                continue

            if len(jsonl_line['passages']) >= 20 and jsonl_line['qid'] == jsonl_line['passages'][0]['docid']:
                filtered_samples.append(jsonl_line)

    print(f'{file_path}\n')
    print(f"Total samples processed: {total_count}\n")
    print(f"Total filtered samples: {len(filtered_samples)}\n")
    ratio = (len(filtered_samples) / total_count) if total_count else 0
    print(f"Ratio of filtered samples to total count: {ratio:.2%}\n")

    return filtered_samples

beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'scidocs', 'climate-fever']

query_types = ['titles', 'claims', 'questions', 'random', 'msmarco', 'keywords']

retrievers = ['nomic-embed-text-v1', 'nomic-embed-text-v1-unsupervised', 'modernbert-embed-base', 'modernbert-embed-base-unsupervised']

for retriever in retrievers:
    for beir_dataset in beir_datasets:
        for query_type in query_types:
            positives_counts = []
            filtered_count = 0

            # Filter samples first
            file_path = f'jsonl_after_reranking/{retriever}_{beir_dataset}-queries-{query_type}.jsonl'
            filtered_samples = filter_samples(file_path)

            # Process filtered samples
            output_file_path = f'outputs/{retriever}_{beir_dataset}-queries-{query_type}.jsonl'
            with open(output_file_path, 'w', encoding='utf-8') as output_f:
                for jsonl_line in tqdm(filtered_samples, desc=f'Processing {file_path}'):
                    output_f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")

