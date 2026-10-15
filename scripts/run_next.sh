#!/usr/bin/env bash
# After the final RAG/export step: d1 LoRA (English, then English + NLLB Arabic), the embedding benchmark,
# then rebuild the exports and the leaderboard.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while systemctl --user is-active -q ticket-final; do sleep 30; done
for d in en en_ar; do .venv/bin/python -u scripts/finetune_d1.py --data $d --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s"; done
.venv/bin/python -u scripts/embed_bench.py --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|Batches"
.venv/bin/python -u scripts/export_predictions.py > results/export.log 2>&1
.venv/bin/python -u scripts/bench_table.py > results/bench_table.log 2>&1
echo NEXT_DONE
