import json
import numpy as np
from tqdm import tqdm
import os

# beir_datasets=['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'scidocs', 'climate-fever']
beir_datasets = ['scifact', 'nfcorpus']

query_types = ['titles', 'claims', 'questions', 'random', 'msmarco', 'keywords']

retrievers = ['nomic-embed-text-v1', 'modernbert-embed-base']

for beir_dataset in beir_datasets:
    for query_type in query_types:
        for retriever in retrievers:
            print(beir_dataset, query_type, retriever)

            scores = []
            old_file_path = f'outputs/{retriever}_{beir_dataset}-queries-{query_type}.jsonl'
            normalized_file_path = f'final_outputs/{retriever}_{beir_dataset}-queries-{query_type}-normalized.jsonl'
            
            with open(old_file_path, 'r', encoding='utf-8') as input_f:
                for line in tqdm(input_f):
                    jsonl_line = json.loads(line)
                    for i in range(len(jsonl_line['passages'])):
                        scores.append(jsonl_line['passages'][i]['score'])

            score_cutoff = np.percentile(scores, 25)
            score_min = np.percentile(scores, 1)
            score_max = np.percentile(scores, 99)

            score_filtered_count = 0

            with open(old_file_path, 'r', encoding='utf-8') as input_f:
                with open(normalized_file_path, 'w', encoding='utf-8') as output_f:
                    for line in tqdm(input_f):
                        jsonl_line = json.loads(line)

                        if jsonl_line['passages'][0]['score'] < score_cutoff:
                            score_filtered_count += 1
                            continue

                        for i in range(len(jsonl_line['passages'])):
                            score = jsonl_line['passages'][i]['score']
                            score = (score - score_min) / (score_max - score_min)
                            score = min(max(score, 0), 1)
                            jsonl_line['passages'][i]['score'] = score
                        jsonl_line['passages'] = jsonl_line['passages'][:20]
                        output_f.write(json.dumps(jsonl_line, ensure_ascii=False) + '\n')

            os.remove(old_file_path)
            
            print("score_filtered_count",  score_filtered_count)