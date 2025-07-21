for dataset in 'nfcorpus' 'scidocs' 'arguana' 'scifact'; do
    model=nomic-ai/nomic-embed-text-v1
    model_name=nomic-embed-text-v1
    model_prefix=nomic-ai

    python encode_corpus.py --model_name ${model} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}

    export TRANSFORMERS_TRUST_REMOTE_CODE=1qq
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output results/run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query \
    --device cuda:0

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    rm -r indices/${model_prefix}_${model_name}_${dataset}_index

    echo "Completed evaluation for dataset: ${dataset}"
done