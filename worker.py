"""Standalone background trading worker for hosted deployments.

This process runs independently of the Streamlit UI and keeps the trading loop
alive as long as the container is up.
"""

from __future__ import annotations

import time

from config import DEFAULT_WATCHLIST, SCAN_INTERVAL_SECONDS
from orchestrator import run_trading_cycle, sync_alpaca_trade_log
from state import APP_STATE


def main() -> None:
    watchlist = DEFAULT_WATCHLIST
    APP_STATE.set_watchlist(watchlist)
    APP_STATE.add_agent_log(
        "Orchestrator",
        "Standalone trading worker started. Background scan loop active.",
        "info",
    )

    while True:
        try:
            run_trading_cycle(watchlist)
            sync_alpaca_trade_log()
        except Exception as exc:  # pragma: no cover
            APP_STATE.add_agent_log(
                "Orchestrator",
                f"Standalone worker error: {exc}",
                "error",
            )

        for _ in range(SCAN_INTERVAL_SECONDS):
            time.sleep(1)


if __name__ == "__main__":
    main()
