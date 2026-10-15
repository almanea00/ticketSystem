#!/usr/bin/env bash
# After the zero-shot benchmark: LoRA-train d1, then Faisal, on the same 487 clean messages, and test again.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while systemctl --user is-active -q ticket-s10k; do sleep 20; done
for w in d1 faisal; do .venv/bin/python -u scripts/support10k_train.py $w --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|falling back"; done
echo TRAIN_DONE
