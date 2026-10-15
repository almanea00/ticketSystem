#!/usr/bin/env bash
# (Re)start the web app as a background user service; it reads .env. Logs: results/web.log
cd "$(dirname "$0")/.."
systemctl --user stop ticket-web 2>/dev/null; systemctl --user reset-failed ticket-web 2>/dev/null
systemd-run --user --unit=ticket-web --working-directory="$PWD" -p ManagedOOMPreference=avoid \
  -p StandardOutput=append:"$PWD/results/web.log" -p StandardError=append:"$PWD/results/web.log" \
  .venv/bin/python -u -m ticketassist.web --port 7861
