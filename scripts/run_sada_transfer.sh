#!/usr/bin/env bash
# d1 LoRA trained ONLY on SADA (dialect ID), then tested on the intent benchmark.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P -u scripts/eval_d1_lora.py --lora runs/d1_lora_sada --name d1_lora_sada150 --title "d1-3B + LoRA on SADA only (150 examples)" --device cuda:0
$P -u scripts/sada_dialect.py --per 500 --epochs 1 --out runs/d1_lora_sada1500 --device cuda:0
$P -u scripts/eval_d1_lora.py --lora runs/d1_lora_sada1500 --name d1_lora_sada1500 --title "d1-3B + LoRA on SADA only (1,500 examples)" --device cuda:0
echo SADA_TRANSFER_DONE
