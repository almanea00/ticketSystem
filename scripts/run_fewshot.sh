#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 1 2; do echo "=== k=$k"; .venv/bin/python -u scripts/eval_d1_fewshot.py --k $k --device cuda:0 2>&1 | grep -E "^===|macro-F1|course|latency|Error|Traceback"; done
echo FEWSHOT_DONE
