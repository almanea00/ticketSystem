#!/usr/bin/env bash
# After the running queue (RAM allows one job at a time): MARBERTv2 on both datasets.
cd "$(dirname "$0")/.."
while systemctl --user is-active -q ticket-queue; do sleep 30; done
.venv/bin/python -u scripts/finetune_marbert.py tobi > results/marbert_tobi.log 2>&1
.venv/bin/python -u scripts/finetune_marbert.py bitext > results/marbert_bitext.log 2>&1
echo MARBERT_DONE >> results/marbert_bitext.log
