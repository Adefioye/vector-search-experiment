for dataset in 'nfcorpus' 'scifact' 'trec-covid' 'fiqa' 'scidocs' 'arguana' 'webis-touche2020' 'climate-fever'; do

    # Change the querie back to 'search_query: ' for consistency
    # Also change --pooling to 'mean' for better performance on these datasets
    # DONE
    # model=nomic-ai/nomic-embed-text-v1
    # model_name=nomic-embed-text-v1
    # model_prefix=nomic-ai

    # model=nomic-ai/nomic-embed-text-v1-unsupervised
    # model_name=nomic-embed-text-v1-unsupervised
    # model_prefix=nomic-ai

    # model=nomic-ai/modernbert-embed-base
    # model_name=modernbert-embed-base
    # model_prefix=nomic-ai

    model=nomic-ai/modernbert-embed-base-unsupervised
    model_name=modernbert-embed-base-unsupervised
    model_prefix=nomic-ai

    python listwise_distillation/encoding/encode_corpus.py --model_name ${model} --normalize --pooling mean --batch_size 1800 --dataset ${dataset}

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