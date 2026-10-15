#!/usr/bin/env bash
# After the main benchmark: validation check of Faisal single-pass-27 vs tournament, then its full benchmark row.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while systemctl --user is-active -q ticket-bench; do sleep 30; done
.venv/bin/python -u scripts/faisal_27.py --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|falling back"
.venv/bin/python -u scripts/bench.py faisal27_zeroshot --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|falling back"
echo F27_DONE
