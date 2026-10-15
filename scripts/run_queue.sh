#!/usr/bin/env bash
# One job at a time: the search website already uses ~11 GB of the 15 GB RAM.
cd "$(dirname "$0")/.."
export USE_TF=0
.venv/bin/python -u scripts/bitext.py prep > results/bitext_prep.log 2>&1 && \
.venv/bin/python -u scripts/bitext.py eval --device cuda:0 --laya "laya-ara,laya (en),laya multilingual" > results/bitext_eval.log 2>&1
echo BITEXT_DONE >> results/bitext_eval.log
.venv/bin/python -u scripts/eval_laya.py --device cuda:0 --models "laya (en),laya typed-decisions,laya multilingual" > results/laya.log 2>&1
echo LAYA_DONE >> results/laya.log
