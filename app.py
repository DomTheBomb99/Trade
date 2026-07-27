"""Streamlit dashboard for the multi-agent Alpaca paper trading system."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import streamlit as st  # type: ignore
import yfinance as yf  # type: ignore

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

AGENT_PERSONAS = {
    "Technical Market Scanner": {
        "name": "Astra",
        "icon": "🛰️",
        "tagline": "Live chart tracking and momentum scouting.",
    },
    "Web Sentiment Analyst": {
        "name": "Nova",
        "icon": "🧠",
        "tagline": "Reading market mood and news signals.",
    },
    "Strategy Selector": {
        "name": "Helix",
        "icon": "🧭",
        "tagline": "Selecting the best approach for the moment.",
    },
    "Risk Manager": {
        "name": "Guardian",
        "icon": "🛡️",
        "tagline": "Protecting cash and enforcing risk rules.",
    },
    "Orchestrator": {
        "name": "Pulse",
        "icon": "⚡",
        "tagline": "Coordinating the live workflow.",
    },
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
        persona = AGENT_PERSONAS.get(entry.agent, {})
        name = persona.get("name", entry.agent)
        ts = entry.timestamp.replace("T", " ").split("+")[0]
        st.markdown(
            f"""
            <div style="
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 12px;
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(96, 165, 250, 0.25);
                backdrop-filter: blur(18px);
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35);
                color: #f8fafc;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:0.75rem;">
                    <div>
                        <strong style="color:{color}; font-size:1rem;">{icon} {name}</strong>
                        <div style="color:#cbd5e1; font-size:0.88rem;">{entry.agent}</div>
                    </div>
                    <span style="color:#94a3b8; font-size:0.82rem;">{ts} UTC</span>
                </div>
                <div style="margin-top:0.75rem; color:#e2e8f0;">{entry.message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _summarize_agent_status(logs: list) -> list[dict[str, str]]:
    status: dict[str, dict[str, str]] = {}
    for entry in logs:
        if entry.agent not in status or entry.timestamp > status[entry.agent]["timestamp"]:
            status[entry.agent] = {
                "agent": entry.agent,
                "message": entry.message,
                "timestamp": entry.timestamp.replace("T", " ").split("+")[0],
                "level": entry.level,
            }
    return [
        {
            "agent": agent,
            "name": AGENT_PERSONAS.get(agent, {}).get("name", agent),
            "icon": AGENT_PERSONAS.get(agent, {}).get("icon", "🤖"),
            "tagline": AGENT_PERSONAS.get(agent, {}).get("tagline", ""),
            "message": details["message"],
            "timestamp": details["timestamp"],
            "level": details["level"],
        }
        for agent, details in status.items()
    ]


def _render_agent_status_cards(status_list: list[dict[str, str]]) -> None:
    if not status_list:
        st.info("Agent status will appear here after the first scan.")
        return

    cols = st.columns(max(1, len(status_list)))
    for idx, status in enumerate(status_list):
        with cols[idx]:
            color = AGENT_COLORS.get(status["agent"], "#64748b")
            st.markdown(
                f"""
                <div style="
                    border-radius: 24px;
                    padding: 22px;
                    margin-bottom: 12px;
                    background: rgba(15, 23, 42, 0.82);
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    backdrop-filter: blur(18px);
                    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.32);
                ">
                    <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:8px;">
                        <span style="font-size:1.4rem;">{status['icon']}</span>
                        <div>
                            <strong style="color:{color}; font-size:1rem;">{status['name']}</strong>
                            <div style="color:#94a3b8; font-size:0.86rem;">{status['agent']}</div>
                        </div>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.9rem; margin-bottom:10px;">{status['tagline']}</div>
                    <div style="color:#e2e8f0; font-size:0.92rem;">{status['message']}</div>
                    <div style="color:#94a3b8; font-size:0.78rem; margin-top:12px;">Last update: {status['timestamp']} UTC</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(ttl=45)
def _fetch_watchlist_prices(symbols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols[:10]:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="1d", interval="1m", actions=False)
            if history.empty:
                raise ValueError("No price data")
            latest = history["Close"].iloc[-1]
            previous = history["Close"].iloc[-2] if len(history) > 1 else latest
            change = latest - previous
            rows.append(
                {
                    "Symbol": symbol,
                    "Price": round(float(latest), 2),
                    "Change": round(float(change), 2),
                    "% Change": f"{round(float(change / previous * 100), 2) if previous else 0:.2f}%",
                    "Volume": int(history["Volume"].iloc[-1]),
                }
            )
        except Exception:
            rows.append(
                {
                    "Symbol": symbol,
                    "Price": "N/A",
                    "Change": "N/A",
                    "% Change": "N/A",
                    "Volume": "N/A",
                }
            )
    return rows


def _render_price_board(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("Current watchlist price data is unavailable right now.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_portfolio(trader: Any) -> None:
    if trader is None:
        st.info("Alpaca not connected. Enter valid credentials to see live portfolio positions.")
        return

    try:
        account = trader.get_account()
        positions = trader.get_positions()
    except Exception as exc:
        st.error(f"Failed to fetch portfolio data: {exc}")
        return

    st.markdown("### Live Portfolio")
    perf_cols = st.columns(3)
    with perf_cols[0]:
        st.metric("Equity", f"${account.equity:,.2f}")
    with perf_cols[1]:
        st.metric("Buying Power", f"${account.buying_power:,.2f}")
    with perf_cols[2]:
        st.metric("Portfolio Value", f"${account.portfolio_value:,.2f}")

    if not positions:
        st.info("No open positions in the Alpaca paper account.")
        return

    portfolio_rows = []
    for pos in positions:
        portfolio_rows.append(
            {
                "Symbol": pos["symbol"],
                "Side": pos["side"],
                "Qty": pos["qty"],
                "Current Price": round(pos["current_price"], 2),
                "Market Value": round(pos["market_value"], 2),
                "P/L": round(pos["unrealized_pl"], 2),
                "P/L %": f"{round(pos["unrealized_plpc"] * 100, 2)}%",
            }
        )
    st.dataframe(portfolio_rows, use_container_width=True, hide_index=True)


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
            strategy_label = decision.get("strategy", "Adaptive")
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
                        Strategy: {strategy_label}
                    </div>
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
        body {
            background: linear-gradient(135deg, #01050f 0%, #09111e 45%, #121c34 100%);
            color: #e2e8f0;
        }
        .glass-card {
            border-radius: 24px;
            padding: 22px;
            margin-bottom: 16px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            backdrop-filter: blur(20px);
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.32);
        }
        .hero-banner {
            border-radius: 28px;
            padding: 26px;
            margin-bottom: 26px;
            background: rgba(22, 28, 45, 0.85);
            border: 1px solid rgba(96, 165, 250, 0.15);
            backdrop-filter: blur(24px);
            box-shadow: 0 26px 70px rgba(15, 23, 42, 0.32);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.16);
            color: #c7d2fe;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .bot-pulse {
            animation: pulse-border 3s ease-in-out infinite;
        }
        @keyframes pulse-border {
            0%, 100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.25); }
            50% { box-shadow: 0 0 0 18px rgba(56, 189, 248, 0.02); }
        }
        .streamlit-expanderHeader {
            color: #dbeafe !important;
        }
        .css-1lsmgbg {
            background: transparent !important;
        }
        .stButton>button {
            border-radius: 999px;
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

    _inject_styles()
    snapshot = APP_STATE.snapshot()

    st.markdown(
        f"""
        <div class='hero-banner'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;'>
                <div>
                    <h1 style='margin:0; font-size:2.6rem;'>Multi-Agent Paper Trading Hub</h1>
                    <p style='margin:12px 0 0; color:#cbd5e1; font-size:1rem;'>
                        Live market scanning, sentiment reading, strategy selection, and risk enforcement — all in a single glossy control center.
                    </p>
                </div>
                <div style='display:flex; gap:1rem; flex-wrap:wrap;'>
                    <div class='glass-card' style='min-width:170px;'>
                        <div style='color:#60a5fa; font-size:0.8rem; letter-spacing:0.08em;'>WATCHLIST</div>
                        <div style='font-size:1.6rem; font-weight:700;'>{len(watchlist)}</div>
                    </div>
                    <div class='glass-card' style='min-width:170px;'>
                        <div style='color:#7dd3fc; font-size:0.8rem; letter-spacing:0.08em;'>ENGINE</div>
                        <div style='font-size:1.6rem; font-weight:700;'>{'Live' if st.session_state.auto_scan else 'Idle'}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    equity_history = snapshot.get("equity_history", [])
    equity_trend = 0.0
    if len(equity_history) >= 2:
        equity_trend = (equity_history[-1] - equity_history[0]) / max(equity_history[0], 1)

    mode_col, strategy_col, winrate_col, trend_col = st.columns(4)
    with mode_col:
        st.metric("Decision Mode", snapshot.get("active_mode", "deterministic").title())
    with strategy_col:
        st.metric("Active Strategy", snapshot.get("active_strategy", "momentum"))
    with winrate_col:
        st.metric("Backtest Win Rate", f"{snapshot.get('win_rate', 0.0) * 100:.1f}%")
    with trend_col:
        st.metric("Equity Trend", f"{equity_trend * 100:.2f}%")

    market_col, portfolio_col = st.columns(2)
    with market_col:
        st.markdown("## Live Price Board")
        _render_price_board(_fetch_watchlist_prices(watchlist))
    with portfolio_col:
        st.markdown("## Live Portfolio Overview")
        trader = create_trader()
        _render_portfolio(trader)

    with st.expander("Strategy Selection Notes", expanded=True):
        st.write(snapshot.get("strategy_reason", "Strategy selection will show after a completed scan."))
        if snapshot.get("strategy_rankings"):
            st.write("**Strategy mix:**")
            for strategy, pct in snapshot["strategy_rankings"].items():
                st.write(f"- {strategy}: {pct:.0%}")
        if snapshot.get("strategy_bias"):
            st.write("**Adaptive strategy bias:**")
            for strategy, weight in snapshot["strategy_bias"].items():
                st.write(f"- {strategy}: x{weight:.2f}")

    with st.expander("Backtest Summary", expanded=False):
        if snapshot["backtest_summary"]:
            st.text(snapshot["backtest_summary"])
        else:
            st.write("Backtest results will populate after the next scan.")

    with st.expander("Latest Scan Summary", expanded=False):
        if snapshot["last_scan_summary"]:
            st.text(snapshot["last_scan_summary"])
        else:
            st.write("Scan summary will appear here once the system completes a cycle.")

    with st.container():
        st.markdown("## Current Agent Status")
        _render_agent_status_cards(_summarize_agent_status(snapshot["agent_logs"]))

    with st.container():
        st.markdown("## Latest Approved Trade Decisions")
        _render_decision_cards(snapshot["decision_summaries"])

    with st.container():
        st.markdown("## Recent Paper Trades")
        _render_trade_logs(snapshot["trade_logs"])

    with st.container():
        st.markdown("## Live Agent Log")
        _render_agent_logs(snapshot["agent_logs"])

    st.caption(
        f"Last UI refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    if st.session_state.auto_scan:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
