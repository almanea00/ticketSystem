#!/usr/bin/env bash
# After everything else on the 3090: re-train the English d1 LoRA so its weights are saved, then the policy/safety benchmark.
cd "$(dirname "$0")/.."
export USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for u in ticket-lora ticket-final; do while systemctl --user is-active -q $u; do sleep 30; done; done
[ -f runs/d1_lora_en/lora.pt ] || .venv/bin/python -u scripts/finetune_d1.py --data en --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s"
.venv/bin/python -u scripts/policy_bench.py --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s|Generating|falling back"
echo POLICY_DONE
