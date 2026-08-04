#!/bin/sh
set -eu

PORT="${PORT:-10000}"

if command -v python >/dev/null 2>&1; then
  python worker.py > /tmp/autotrader-worker.log 2>&1 &
elif command -v python3 >/dev/null 2>&1; then
  python3 worker.py > /tmp/autotrader-worker.log 2>&1 &
else
  echo "No Python interpreter available in container" >&2
  exit 1
fi

exec streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
