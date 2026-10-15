#!/usr/bin/env bash
# When ALLaM's dialect sets are done (3060 free): RAG retrieval test, then score every finished benchmark
# method on the extra test sets (course Arabic tickets + Hijazi / Gulf / Najdi rewrites). No latency here.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while systemctl --user is-active -q ticket-dialect; do sleep 30; done
.venv/bin/python -u scripts/rag_eval.py --device cuda:1 > results/rag.log 2>&1
for m in tfidf_lr e5_lr embgemma_lr nli_zeroshot laya_ml_zeroshot laya_ara_zeroshot d1_zeroshot distilmbert_ft marbert_ft qwen05_lora llm_fewshot faisal_zeroshot faisal27_zeroshot; do
  while [ ! -f results/bench/$m.json ]; do sleep 60; done   # wait until the main benchmark has run it
  echo "=== $m"; .venv/bin/python -u scripts/bench.py $m --device cuda:1 --extra-only 2>&1 | grep -E "^extra|Error"
done
echo EXTRA_DONE
