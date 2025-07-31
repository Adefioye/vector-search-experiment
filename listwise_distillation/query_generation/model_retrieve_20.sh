dataset='msmarco'

model=nomic-ai/nomic-embed-text-v1
model_name=nomic-sup

# model=nomic-ai/nomic-embed-text-v1-unsupervised
# model_name=nomic-unsup

# model=nomic-ai/modernbert-embed-base
# model_name=modernbert-sup

# model=nomic-ai/modernbert-embed-base-unsupervised
# model_name=modernbert-unsup

python listwise_distillation/encoding/encode_corpus.py --model_name ${model} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}

for query_type in 'keywords' 'titles' 'claims' 'questions' 'random' 'msmarco'; do
    python -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder ${model} --l2-norm --query-prefix "search_query: " \
      --index indices/${model_name}_${dataset}_index \
      --topics generated_queries/${dataset}_generated_queries_${query_type}.tsv \
      --output retrieval_runs/run.${model_name}.${dataset}.generated-queries-${query_type}_20.txt \
      --hits 20 \
      --device cuda:0
done


# This helps produce retrieval trec runs for each query type in the dataset.
for dataset in 'fiqa' 'scifact' 'trec-covid' 'nfcorpus' 'arguana' 'webis-touche2020' 'scidocs' 'climate-fever'; do
  # This is to encode the corpus for each dataset
  python listwise_distillation/encoding/encode_corpus.py --model_name ${model} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}

  for query_type in 'keywords' 'titles' 'claims' 'questions' 'random' 'msmarco'; do

    python -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder ${model} --l2-norm --query-prefix "search_query: " \
      --index indices/${model_name}_${dataset}_index \
      --topics generated_queries/${dataset}_generated_queries_${query_type}.tsv \
      --output retrieval_runs/run.${model_name}.${dataset}.generated-queries-${query_type}_20.txt \
      --hits 20 \
      --device cuda:0
  done
done