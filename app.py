"""Streamlit dashboard for the multi-agent Alpaca paper trading system."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import streamlit as st

from config import DEFAULT_WATCHLIST, SCAN_INTERVAL_SECONDS
from orchestrator import run_trading_cycle, sync_alpaca_trade_log
from state import APP_STATE
from trading import create_trader

st.set_page_config(
    page_title="Multi-Agent Paper Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENT_COLORS = {
    "Technical Market Scanner": "#2563eb",
    "Web Sentiment Analyst": "#7c3aed",
    "Risk Manager": "#dc2626",
    "Orchestrator": "#059669",
}

LEVEL_ICONS = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
}


def _init_session() -> None:
    if "watchlist_text" not in st.session_state:
        st.session_state.watchlist_text = ", ".join(DEFAULT_WATCHLIST)
    if "auto_scan" not in st.session_state:
        st.session_state.auto_scan = False
    if "scan_thread" not in st.session_state:
        st.session_state.scan_thread = None
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()


def _parse_watchlist(text: str) -> list[str]:
    return [s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()]


def _background_scan_loop(watchlist: list[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        run_trading_cycle(watchlist)
        sync_alpaca_trade_log()
        for _ in range(SCAN_INTERVAL_SECONDS):
            if stop_event.is_set():
                break
            time.sleep(1)


def _start_auto_scan(watchlist: list[str]) -> None:
    stop_event: threading.Event = st.session_state.stop_event
    stop_event.set()
    if st.session_state.scan_thread and st.session_state.scan_thread.is_alive():
        st.session_state.scan_thread.join(timeout=2)
    stop_event.clear()
    thread = threading.Thread(
        target=_background_scan_loop,
        args=(watchlist, stop_event),
        daemon=True,
    )
    st.session_state.scan_thread = thread
    thread.start()


def _stop_auto_scan() -> None:
    st.session_state.stop_event.set()


def _render_agent_logs(logs: list) -> None:
    if not logs:
        st.info("No agent activity yet. Run a scan to populate the live log.")
        return

    for entry in logs[:80]:
        color = AGENT_COLORS.get(entry.agent, "#64748b")
        icon = LEVEL_ICONS.get(entry.level, "ℹ️")
        ts = entry.timestamp.replace("T", " ").split("+")[0]
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color};
                padding: 0.55rem 0.85rem;
                margin-bottom: 0.45rem;
                background: #f8fafc;
                border-radius: 0 6px 6px 0;
            ">
                <small style="color:#64748b;">{ts} UTC</small><br/>
                <strong style="color:{color};">{icon} {entry.agent}</strong><br/>
                <span>{entry.message}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_trade_logs(logs: list) -> None:
    if not logs:
        st.info("No paper trades logged yet.")
        return

    rows = []
    for entry in logs[:40]:
        rows.append(
            {
                "Time (UTC)": entry.timestamp.replace("T", " ").split("+")[0],
                "Symbol": entry.symbol,
                "Side": entry.side.upper(),
                "Qty": entry.qty,
                "Status": entry.status,
                "Order ID": entry.order_id[:8] + "...",
                "Details": entry.details,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    _init_session()

    st.title("Multi-Agent Paper Trading System")
    st.caption("CrewAI expert team · Alpaca Paper Trading · yfinance + news sentiment")

    with st.sidebar:
        st.header("Watchlist")
        watchlist_text = st.text_area(
            "Enter tickers (comma or newline separated)",
            value=st.session_state.watchlist_text,
            height=120,
            help="High-volume tech names work best for the momentum scanner.",
        )
        st.session_state.watchlist_text = watchlist_text
        watchlist = _parse_watchlist(watchlist_text)
        APP_STATE.set_watchlist(watchlist)

        st.divider()
        st.subheader("Controls")

        if st.button("Run Scan Now", type="primary", use_container_width=True):
            with st.spinner("Agents collaborating..."):
                run_trading_cycle(watchlist)
                sync_alpaca_trade_log()
            st.success("Scan complete.")

        auto_scan = st.toggle("Auto-scan every 5 minutes", value=st.session_state.auto_scan)
        if auto_scan and not st.session_state.auto_scan:
            st.session_state.auto_scan = True
            _start_auto_scan(watchlist)
        elif not auto_scan and st.session_state.auto_scan:
            st.session_state.auto_scan = False
            _stop_auto_scan()

        if st.button("Refresh Alpaca Orders", use_container_width=True):
            sync_alpaca_trade_log()

        st.divider()
        st.subheader("Team")
        st.markdown(
            """
            - **Technical Market Scanner** — momentum, MAs, RSI, MACD
            - **Web Sentiment Analyst** — Yahoo Finance + Google News
            - **Risk Manager** — 2% max risk, 3:1 R:R, tight stops
            """
        )

        trader = create_trader()
        if trader:
            try:
                account = trader.get_account()
                st.metric("Paper Equity", f"${account.equity:,.2f}")
                st.metric("Buying Power", f"${account.buying_power:,.2f}")
            except Exception as exc:
                st.error(f"Alpaca error: {exc}")
        else:
            st.warning("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in Render env vars.")

    snapshot = APP_STATE.snapshot()

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("Watchlist Size", len(watchlist))
    with status_col2:
        scanning = "Scanning..." if snapshot["is_scanning"] else "Idle"
        st.metric("Engine Status", scanning)
    with status_col3:
        st.metric("Agent Log Entries", len(snapshot["agent_logs"]))

    if snapshot["last_error"]:
        st.error(snapshot["last_error"])

    if snapshot["last_scan_summary"]:
        with st.expander("Latest Crew / Scan Summary", expanded=False):
            st.text(snapshot["last_scan_summary"])

    log_col, trade_col = st.columns([1.1, 1])

    with log_col:
        st.subheader("Live Agent Log")
        log_container = st.container(height=520)
        with log_container:
            _render_agent_logs(snapshot["agent_logs"])

    with trade_col:
        st.subheader("Recent Paper Trades")
        _render_trade_logs(snapshot["trade_logs"])

    st.caption(
        f"Last UI refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    if st.session_state.auto_scan:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
