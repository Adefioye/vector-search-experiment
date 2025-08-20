#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting all query generation scripts at $(date)"

# echo
# echo "1️⃣  Running claims_gen_vllm.py..."
# python listwise_distillation/query_generation/query_gen_scripts/claims_gen_vllm.py
# echo "✅ claims_gen_vllm.py finished at $(date)"

# echo
# echo "2️⃣  Running keywords_gen_vllm.py..."
# python listwise_distillation/query_generation/query_gen_scripts/keywords_gen_vllm.py
# echo "✅ keywords_gen_vllm.py finished at $(date)"

echo
echo "3️⃣  Running msmarco_query_gen_vllm.py..."
python listwise_distillation/query_generation/query_gen_scripts/msmarco_query_gen_vllm.py
echo "✅ msmarco_query_gen_vllm.py finished at $(date)"

echo
echo "4️⃣  Running question_gen_vllm.py..."
python listwise_distillation/query_generation/query_gen_scripts/question_gen_vllm.py
echo "✅ question_gen_vllm.py finished at $(date)"

echo
echo "5️⃣  Running random_query_gen_vllm.py..."
python listwise_distillation/query_generation/query_gen_scripts/random_query_gen_vllm.py
echo "✅ random_query_gen_vllm.py finished at $(date)"

echo
echo "6️⃣  Running title_gen_vllm.py..."
python listwise_distillation/query_generation/query_gen_scripts/title_gen_vllm.py
echo "✅ title_gen_vllm.py finished at $(date)"

echo
echo "🏁 All query generation scripts completed at $(date)"
