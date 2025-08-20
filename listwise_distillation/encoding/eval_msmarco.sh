dataset=msmarco
model_prefix=nomic-ai

for model_name in 'nomic-embed-text-v1'; do

    # Searching for d19
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl19
    echo "Evaluating run.${model_name}.dl19.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  

    # Searching for d20
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl20
    echo "Evaluating run.${model_name}.dl20.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  

    # Echo retrieval and evaluation completion
    echo "✅ Retrieval and evaluation completed for model: $model_name on dataset: $dataset"
done

for model_name in 'nomic-embed-text-v1-unsupervised'; do

    # Searching for d19
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl19
    echo "Evaluating run.${model_name}.dl19.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  

    # Searching for d20
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl20
    echo "Evaluating run.${model_name}.dl20.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  

    # Echo retrieval and evaluation completion
    echo "✅ Retrieval and evaluation completed for model: $model_name on dataset: $dataset"
done

for model_name in 'modernbert-embed-base'; do

    # Searching for d19
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl19
    echo "Evaluating run.${model_name}.dl19.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  

    # Searching for d20
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl20
    echo "Evaluating run.${model_name}.dl20.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  

    # Echo retrieval and evaluation completion
    echo "✅ Retrieval and evaluation completed for model: $model_name on dataset: $dataset"
done

for model_name in 'modernbert-embed-base-unsupervised'; do

    # Searching for d19
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl19-passage \
    --output run.${model_name}.dl19.txt \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl19
    echo "Evaluating run.${model_name}.dl19.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.${model_name}.dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.${model_name}.dl19.txt  

    # Searching for d20
    echo "Retrieving for model: $model_name on dataset: $dataset"
    python -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto \
    --encoder ${model_prefix}/${model_name} --l2-norm --query-prefix "search_query: " \
    --index indices/${model_prefix}_${model_name}_${dataset}_index \
    --topics dl20 \
    --output run.${model_name}.dl20.txt  \
    --hits 1000 \
    --device cuda:0

    # echo evaluation for dl20
    echo "Evaluating run.${model_name}.dl20.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.${model_name}.dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.${model_name}.dl20.txt  

    # Echo retrieval and evaluation completion
    echo "✅ Retrieval and evaluation completed for model: $model_name on dataset: $dataset"
done