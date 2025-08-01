import json
from tqdm import tqdm
from datasets import load_dataset

beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'climate-fever']
query_types = ['titles', 'claims', 'questions', 'random', 'msmarco', 'keywords']
retrievers = ['nomic-embed-text-v1', 'nomic-embed-text-v1-unsupervised', 'modernbert-embed-base', 'modernbert-embed-base-unsupervised']

for retriever in retrievers:
    for beir_dataset in beir_datasets:
        passages_dict = {}
        huggingface_passages = load_dataset('BeIR/' + beir_dataset, 'corpus')['corpus']
        for passage in tqdm(huggingface_passages):
            passages_dict[str(passage['_id'])] = passage['text'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]

        
        for query_type in query_types:
            queries_dic = {}
            with open('generated_queries/' + beir_dataset + '_generated_queries_' + query_type + '.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    vals = line.split('\t')
                    queries_dic[vals[0]] = vals[1]

            rankings_dict = {}
            # retrieval_runs/run.{retriever}.{beir_dataset}.generated-queries-{query_type}.filtered.txt
            with open('retrieval_runs/run.' + retriever + '.' + beir_dataset + '.' + 'generated-queries-' + query_type + '.filtered.txt', 'r') as f:    
                for line in tqdm(f):
                    vals = line.split(' ')
                    qid = vals[0]
                    pid = vals[2]
                    rank  = int(vals[3])
                    score = float(vals[4])

                    passage_text = passages_dict[str(pid)]

                    if qid in rankings_dict:
                        rankings_dict[qid]['passages'].append({"docid": pid, "text": passage_text, "score": score})
                    else:
                        try: 
                            query_text = queries_dic[qid]   
                            rankings_dict[qid] = {'query': query_text, 'passages':[{"docid": pid, "text": passage_text, "score": score}]}
                        except:
                            continue

            # Let the jsonl_before_reranking file have a retriever component, together with the dataset and query type
            with open('jsonl_before_reranking/' + retriever + '_' + beir_dataset + '-queries-' + query_type +'.jsonl', 'w', encoding='utf-8') as f:
                skipped_count = 0
                for qid in rankings_dict:
                    query_entry = rankings_dict[qid]
                    query_entry['qid'] = qid
                    query_entry['passages'] = query_entry['passages']
                    passages_contains_relevant_passage = False
                    for passage in query_entry['passages']:
                        if passage['docid'] == qid:
                            passages_contains_relevant_passage = True
                    assert(passages_contains_relevant_passage)
                    if (len(query_entry['passages']) < 20):
                        skipped_count+=1
                        continue
                    f.write(json.dumps(query_entry, ensure_ascii=False) + "\n")
                print(retriever, beir_dataset, query_type, skipped_count)

