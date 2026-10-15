#!/usr/bin/env bash
# d1 LoRA first (as soon as the 3090 is free of the speed test), then the embedding benchmark.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while systemctl --user is-active -q ticket-d1speed; do sleep 20; done
for d in en en_ar; do .venv/bin/python -u scripts/finetune_d1.py --data $d --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s"; done
echo LORA_DONE
.venv/bin/python -u scripts/embed_bench.py --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|Batches"
echo EMB_DONE
