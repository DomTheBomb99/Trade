#!/usr/bin/env sh
set -eu

PORT="${PORT:-10000}"

python worker.py > /tmp/autotrader-worker.log 2>&1 &

exec streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
