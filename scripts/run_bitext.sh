#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export USE_TF=0
.venv/bin/python -u scripts/bitext.py prep > results/bitext_prep.log 2>&1 && \
.venv/bin/python -u scripts/bitext.py eval --device cuda:0 > results/bitext_eval.log 2>&1
echo BITEXT_DONE >> results/bitext_eval.log
