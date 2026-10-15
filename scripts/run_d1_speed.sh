#!/usr/bin/env bash
# d1 speed test on the 3090 once nothing else is running on it (the Faisal 27 benchmark row first).
cd "$(dirname "$0")/.."
while systemctl --user is-active -q ticket-f27; do sleep 30; done
.venv/bin/python -u scripts/d1_speed.py --device cuda:0 2>&1 | grep -v -i -E "warn|Loading|it/s"
echo SPEED_DONE
