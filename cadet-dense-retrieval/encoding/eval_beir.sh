for dataset in 'trec-covid' 'nfcorpus' 'fiqa' 'scidocs' 'arguana' 'webis-touche2020'  'dpbedia-entity' 'climate-fever' 'scifact' 'cqadupstack'; do
    model=nomic-ai/modernbert-embed-base
    model_name=modernbert-embed-base
    model_prefix=nomic-ai

    python encode_corpus.py --model_name ${model} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output results/run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    rm -r indices/${model_prefix}_${model_name}_${dataset}_index

    model=nomic-ai/modernbert-embed-base-unsupervised
    model_name=modernbert-embed-base-unsupervised

    python encode_corpus.py --model_name ${model} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output results/run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    rm -r indices/${model_prefix}_${model_name}_${dataset}_index

    model=nomic-ai/nomic-embed-text-v1-unsupervised
    model_name=nomic-embed-text-v1-unsupervised

    python encode_corpus.py --model_name ${model} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output results/run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    rm -r indices/${model_prefix}_${model_name}_${dataset}_index

    model=nomic-ai/nomic-embed-text-v1
    model_name=nomic-embed-text-v1

    python3 encode_corpus.py --model_name ${model} --normalize --pooling cls --batch_size 1800 --dataset ${dataset}

    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics beir-v1.0.0-${dataset}-test \
    --output results/run.beir.${model_name}.${dataset}.txt \
    --hits 1000 --remove-query

    python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    python -m pyserini.eval.trec_eval \
    -c -m recall.100 beir-v1.0.0-${dataset}-test \
    results/run.beir.${model_name}.${dataset}.txt

    rm -r indices/${model_prefix}_${model_name}_${dataset}_index
done