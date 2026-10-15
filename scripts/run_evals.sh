#!/usr/bin/env bash
# Classification, then retrieval tests, on the 3090 (CUDA 0 = fastest GPU).
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -u scripts/eval_classify.py > results/classify.log 2>&1
.venv/bin/python -u scripts/eval_retrieve.py --rerank 300 > results/retrieve.log 2>&1
echo EVALS_DONE >> results/retrieve.log
