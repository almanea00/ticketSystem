#!/usr/bin/env bash
# Every method on the same GPU (the 3090), one at a time, so latencies are comparable.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in tfidf_lr e5_lr embgemma_lr nli_zeroshot laya_ml_zeroshot laya_ara_zeroshot d1_zeroshot distilmbert_ft marbert_ft qwen05_lora llm_fewshot faisal_zeroshot; do
  [ -f results/bench/$m.json ] && continue
  echo "=== $m $(date +%T)"
  .venv/bin/python -u scripts/bench.py $m --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|Batches|fast path|falling back"
done
echo BENCH_DONE
