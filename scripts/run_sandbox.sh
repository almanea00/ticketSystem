#!/usr/bin/env bash
# Banking77 showcase: translate, score d1, then serve the sandbox (d1 stays loaded on the 3090).
cd "$(dirname "$0")/.."
[ -f data/banking77_showcase.parquet ] || .venv/bin/python -u scripts/banking77.py prep > results/banking77_prep.log 2>&1
.venv/bin/python -u scripts/banking77.py eval --device cuda:0 > results/banking77_eval.log 2>&1
exec .venv/bin/python -u -m ticketassist.sandbox --device cuda:0 --port 7862 > results/sandbox.log 2>&1
