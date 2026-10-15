#!/usr/bin/env bash
# Last: RAG with latency on the quiet 3090, then export predictions and build the tables.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for u in ticket-after ticket-d1speed ticket-lora ticket-sada; do while systemctl --user is-active -q $u; do sleep 30; done; done
.venv/bin/python -u scripts/rag_eval.py --device cuda:0 > results/rag.log 2>&1
.venv/bin/python -u scripts/export_predictions.py > results/export.log 2>&1
.venv/bin/python -u scripts/bench_table.py > results/bench_table.log 2>&1
echo FINAL_DONE >> results/bench_table.log
