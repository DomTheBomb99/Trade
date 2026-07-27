"""Streamlit dashboard for the multi-agent Alpaca paper trading system."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import streamlit as st  # type: ignore

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
    "Strategy Selector": "#eab308",
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


def _render_decision_cards(decisions: list[dict[str, Any]]) -> None:
    if not decisions:
        st.info("No approved trade decisions yet. Run a scan to surface candidate trades.")
        return

    cols = st.columns(min(3, max(1, len(decisions))))
    for idx, decision in enumerate(decisions):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    border-radius: 18px;
                    padding: 18px;
                    margin-bottom: 16px;
                    background: #ffffff;
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
                ">
                    <strong style="font-size: 1rem;">{decision['symbol']} · {decision['side'].upper()}</strong>
                    <div style="color:#475569; font-size:0.92rem; margin:8px 0;">
                        Qty: {decision['qty']} · Entry ${decision['entry_price']:.2f}
                    </div>
                    <div style="color:#475569; font-size:0.92rem;">
                        TP ${decision['take_profit']:.2f} · SL ${decision['stop_loss']:.2f}
                    </div>
                    <div style="color:#475569; font-size:0.92rem; margin-top:0.6rem;">
                        R:R {decision['risk_reward_ratio']:.2f} · Trailing {decision['trailing_stop_pct']*100:.2f}%
                    </div>
                    <div style="color:#0f172a; font-size:0.9rem; margin-top:0.85rem;">
                        {decision['rationale']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .agent-card {
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        .agent-card strong {
            color: #0f172a;
            font-size: 1rem;
        }
        .agent-card .agent-meta {
            color: #475569;
            font-size: 0.92rem;
            margin-top: 8px;
        }
        .agent-status-header {
            font-weight: 700;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summarize_agent_activity(logs: list) -> dict[str, dict[str, Any]]:
    activity: dict[str, dict[str, Any]] = {}
    for entry in logs:
        if entry.agent not in activity:
            activity[entry.agent] = {
                "agent": entry.agent,
                "last_message": entry.message,
                "last_time": entry.timestamp.replace("T", " ").split("+")[0],
                "count": 0,
                "level": entry.level,
            }
        activity[entry.agent]["count"] += 1
    return activity


def _summarize_trade_performance(logs: list) -> dict[str, int]:
    summary = {
        "total": len(logs),
        "submitted": 0,
        "filled": 0,
        "canceled": 0,
        "others": 0,
    }
    for entry in logs:
        status = entry.status.lower()
        if status in {"filled", "partial", "partially_filled"}:
            summary["filled"] += 1
        elif status in {"new", "accepted", "submitted", "accepted", "partially_filled"}:
            summary["submitted"] += 1
        elif status in {"canceled", "cancelled", "rejected", "expired"}:
            summary["canceled"] += 1
        else:
            summary["others"] += 1
    return summary


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

        auto_scan = st.checkbox("Auto-scan every 5 minutes", value=st.session_state.auto_scan)
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
            - **Web Sentiment Analyst** — real-time news sentiment and narrative risk
            - **Strategy Selector** — chooses breakout, trend-following, or defensive strategy
            - **Risk Manager** — enforces 2% equity risk, 3:1 R:R, and bracket order discipline
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

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        st.metric("Watchlist Size", len(watchlist))
    with status_col2:
        scanning = "Scanning..." if snapshot["is_scanning"] else "Idle"
        st.metric("Engine Status", scanning)
    with status_col3:
        st.metric("Agent Log Entries", len(snapshot["agent_logs"]))
    with status_col4:
        trade_summary = _summarize_trade_performance(snapshot["trade_logs"])
        st.metric("Paper Trades", trade_summary["total"])

    if snapshot["last_error"]:
        st.error(snapshot["last_error"])

    if snapshot["last_scan_summary"]:
        with st.expander("Latest Crew / Scan Summary", expanded=False):
            st.text(snapshot["last_scan_summary"])

    if snapshot["backtest_summary"]:
        with st.expander("Recent Backtest Summary", expanded=False):
            st.text(snapshot["backtest_summary"])

    _inject_styles()
    agent_activity = _summarize_agent_activity(snapshot["agent_logs"])
    with st.container():
        st.markdown("## Current Agent Activity")
        for agent_name, agent_data in agent_activity.items():
            color = AGENT_COLORS.get(agent_name, "#475569")
            st.markdown(
                f"""
                <div class='agent-card'>
                    <strong style='color:{color};'>{agent_data['agent']}</strong>
                    <div class='agent-meta'>Last updated: {agent_data['last_time']} · Messages: {agent_data['count']} · Level: {agent_data['level']}</div>
                    <div style='margin-top:8px;'>{agent_data['last_message']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    log_col, trade_col = st.columns([1.1, 1])

    with log_col:
        st.subheader("Live Agent Log")
        log_container = st.container()
        with log_container:
            _render_agent_logs(snapshot["agent_logs"])

    with trade_col:
        st.subheader("Recent Paper Trades")
        _render_trade_logs(snapshot["trade_logs"])

    decision_section = st.container()
    with decision_section:
        st.subheader("Latest Approved Trade Decisions")
        _render_decision_cards(snapshot["decision_summaries"])

    st.caption(
        f"Last UI refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    if st.session_state.auto_scan:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
