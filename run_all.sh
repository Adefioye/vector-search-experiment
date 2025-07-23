# Create trec results per model
echo "Running trec results generation for each model in the list..."
chmod +x listwise_distillation/encoding/test_bge_beir.sh
bash listwise_distillation/encoding/test_bge_beir.sh

# Create json/jsonl for queries and corpus file for BEIR datasets
echo "Generating JSON/JSONL files for queries and corpus..."
chmod +x listwise_distillation/reranker/run_generate_beir_json.sh
bash listwise_distillation/reranker/run_generate_beir_json.sh  

# Convert initial trec results to jsonl format
echo "Converting initial TREC results to JSONL format..."
python listwise_distillation/reranker/run_trec_to_jsonl.py

# Run reranking on the generated JSONL files
echo "Running reranking on the generated JSONL files..."
chmod +x listwise_distillation/reranker/run_reranking.sh
bash listwise_distillation/reranker/run_reranking.sh

# Convert reranked results to JSONL format
echo "Converting reranked results to JSONL format..."
python listwise_distillation/reranker/run_jsonl_to_trec.py

# Evaluate the reranked results
echo "Evaluating reranked results..."
chmod +x listwise_distillation/reranker/rerank_eval.sh
bash listwise_distillation/reranker/rerank_eval.sh

# Final message
echo "All steps completed successfully!"