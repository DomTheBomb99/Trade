"""Trading orchestration: CrewAI deliberation + Alpaca execution."""

from __future__ import annotations

import traceback

from config import DEFAULT_WATCHLIST, CREWAI_FORCE_DETERMINISTIC, CREWAI_LLM_TYPE, OPENAI_API_KEY
from state import APP_STATE
from trading import AlpacaTrader, create_trader
from backtest import average_backtest_win_rate, run_backtest, summarize_backtest


def _log(agent: str, message: str, level: str = "info") -> None:
    APP_STATE.add_agent_log(agent, message, level)


def run_trading_cycle(watchlist: list[str] | None = None) -> str:
    """Execute one full multi-agent scan, deliberation, and optional trade cycle."""
    from agents import CREWAI_AVAILABLE, build_trading_crew, extract_crew_output, run_deterministic_pipeline

    symbols = watchlist or APP_STATE.get_watchlist() or DEFAULT_WATCHLIST
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    APP_STATE.set_watchlist(symbols)
    APP_STATE.set_scanning(True)
    APP_STATE.set_last_error("")

    _log("Orchestrator", f"Starting scan for watchlist: {', '.join(symbols)}")

    trader = create_trader()
    if trader is None:
        msg = "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        APP_STATE.set_last_error(msg)
        _log("Orchestrator", msg, "error")
        APP_STATE.set_scanning(False)
        return msg

    try:
        account = trader.get_account()
        APP_STATE.add_equity_point(account.equity)
        equity_trend = APP_STATE.get_equity_trend()
        _log(
            "Orchestrator",
            f"Paper account equity ${account.equity:,.2f} | buying power ${account.buying_power:,.2f} | equity trend {equity_trend*100:.2f}%",
        )

        market_open = trader.is_market_open()
        if not market_open:
            msg = "Market is closed. Analysis completed; orders skipped until market open."
            _log("Orchestrator", msg, "warning")
        else:
            _log("Orchestrator", "Market is open. Trade execution enabled.")

        _log("Technical Market Scanner", "Scanning watchlist for momentum, MAs, RSI, MACD...")
        _log("Web Sentiment Analyst", "Scraping Yahoo Finance + Google News headlines...")

        use_crew = (
            not CREWAI_FORCE_DETERMINISTIC
            and CREWAI_AVAILABLE
            and (CREWAI_LLM_TYPE != "openai" or bool(OPENAI_API_KEY))
        )

        if CREWAI_FORCE_DETERMINISTIC:
            _log(
                "Orchestrator",
                "Deterministic-only mode enabled. Skipping CrewAI and using internal strategy logic.",
                "warning",
            )
            APP_STATE.set_active_mode("deterministic")
            APP_STATE.set_last_scan_summary("Deterministic mode: CrewAI skipped.")
        elif CREWAI_LLM_TYPE == "openai" and not OPENAI_API_KEY:
            _log(
                "Orchestrator",
                "OpenAI API key missing. Skipping CrewAI and using deterministic pipeline.",
                "warning",
            )
            APP_STATE.set_active_mode("deterministic")
            APP_STATE.set_last_scan_summary("No OpenAI key; deterministic pipeline only.")

        if use_crew:
            crew = build_trading_crew(symbols, account)
            if crew is not None:
                APP_STATE.set_active_mode("crew + deterministic")
                _log("Orchestrator", "CrewAI team collaborating on trade decision...")
                try:
                    crew_result = crew.kickoff()
                    crew_summary = extract_crew_output(crew_result)
                    APP_STATE.set_last_scan_summary(crew_summary[:4000])
                    _log("Orchestrator", "Crew deliberation complete. Parsing risk-approved setups...")
                    _log("Risk Manager", crew_summary[:500] + ("..." if len(crew_summary) > 500 else ""))
                except Exception as crew_error:
                    APP_STATE.set_active_mode("deterministic")
                    _log(
                        "Orchestrator",
                        f"CrewAI deliberation failed ({crew_error}). Falling back to deterministic pipeline.",
                        "warning",
                    )
                    APP_STATE.set_last_scan_summary(f"Crew fallback: {crew_error}")
            else:
                APP_STATE.set_active_mode("deterministic")
                _log(
                    "Orchestrator",
                    "CrewAI is unavailable. Using deterministic strategy pipeline only.",
                    "warning",
                )
                APP_STATE.set_last_scan_summary("CrewAI unavailable; deterministic pipeline only.")
        else:
            APP_STATE.set_active_mode("deterministic")
            _log(
                "Orchestrator",
                "Skipping CrewAI. Using deterministic strategy pipeline only.",
                "warning",
            )
            if not APP_STATE.last_scan_summary:
                APP_STATE.set_last_scan_summary("CrewAI skipped; deterministic pipeline only.")

        open_positions = trader.get_open_position_symbols()
        approved_decisions = run_deterministic_pipeline(symbols, account, open_positions)
        APP_STATE.set_decision_summaries(approved_decisions)

        backtest_results = run_backtest(symbols)
        win_rate = average_backtest_win_rate(backtest_results)
        APP_STATE.set_win_rate(win_rate)
        APP_STATE.set_backtest_summary(summarize_backtest(backtest_results))

        equity_trend = APP_STATE.get_equity_trend()
        if win_rate >= 0.55 and equity_trend > 0.01:
            bias = {"Momentum": 1.2, "Trend-Following": 1.1, "Defensive": 0.8}
            bias_reason = "Strong performance; biasing toward growth-oriented strategies."
        elif win_rate < 0.45 or equity_trend < -0.02:
            bias = {"Momentum": 0.8, "Trend-Following": 0.9, "Defensive": 1.3}
            bias_reason = "Weak recent performance; biasing toward defensive risk control."
        else:
            bias = {"Momentum": 1.0, "Trend-Following": 1.0, "Defensive": 1.0}
            bias_reason = "Balanced performance; no strong strategy bias."
        APP_STATE.set_strategy_bias(bias)
        APP_STATE.set_strategy_reason(
            f"{bias_reason} Equity trend {equity_trend*100:.1f}%, backtest win rate {win_rate*100:.1f}%."
        )

        if not approved_decisions:
            APP_STATE.set_decision_summaries([])
            _log("Risk Manager", "No trades approved. Capital preserved.", "warning")
            summary = "Scan complete. No risk-approved trades."
            APP_STATE.set_last_scan_summary(summary)
            APP_STATE.set_scanning(False)
            return summary

        executed_count = 0
        for decision in approved_decisions:
            _log(
                "Technical Market Scanner",
                f"{decision.symbol}: momentum setup identified at ${decision.entry_price:.2f}",
            )
            _log(
                "Web Sentiment Analyst",
                f"{decision.symbol}: sentiment supports {decision.side.upper()} bias",
            )
            _log("Risk Manager", decision.rationale)

            if not market_open:
                _log(
                    "Orchestrator",
                    f"Would execute {decision.side.upper()} {decision.qty} {decision.symbol} at market open",
                )
                continue

            result = trader.execute_bracket_trade(decision)
            if result.get("executed"):
                executed_count += 1
                APP_STATE.add_trade_log(
                    symbol=decision.symbol,
                    side=decision.side,
                    qty=decision.qty,
                    status=result.get("status", "submitted"),
                    order_id=result["entry_order_id"],
                    details=(
                        f"Bracket TP ${result['take_profit']:.2f}, SL ${result['stop_loss']:.2f}, "
                        f"trailing {result['trailing_stop_pct']}%"
                    ),
                )
                _log(
                    "Orchestrator",
                    f"Executed {decision.side.upper()} {decision.qty} {decision.symbol} "
                    f"(order {result['entry_order_id']})",
                )
            else:
                _log("Orchestrator", f"Order skipped for {decision.symbol}: {result.get('reason')}", "warning")

        summary = (
            f"Scan complete. {len(approved_decisions)} approved, {executed_count} executed."
        )
        APP_STATE.set_last_scan_summary(summary)
        APP_STATE.set_scanning(False)
        return summary

    except Exception as exc:
        tb = traceback.format_exc()
        APP_STATE.set_last_error(str(exc))
        _log("Orchestrator", f"Cycle failed: {exc}", "error")
        APP_STATE.set_scanning(False)
        return tb


def sync_alpaca_trade_log(trader: AlpacaTrader | None = None) -> None:
    """Refresh recent Alpaca orders into the dashboard trade log."""
    active_trader = trader or create_trader()
    if active_trader is None:
        return
    try:
        orders = active_trader.get_recent_orders(limit=20)
        entries = []
        for order in orders:
            entries.append(
                (
                    order["symbol"],
                    order["side"],
                    order["filled_qty"] or order["qty"],
                    order["status"],
                    order["id"],
                    f"{order['type']} @ avg ${order['filled_avg_price']:.2f}",
                )
            )
        APP_STATE.replace_trade_logs(entries)
    except Exception as exc:
        APP_STATE.set_last_error(f"Failed to sync Alpaca orders: {exc}")
