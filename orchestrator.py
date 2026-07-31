"""Trading orchestration: CrewAI deliberation + Alpaca execution."""

from __future__ import annotations

import traceback

from config import DEFAULT_WATCHLIST, CREWAI_FORCE_DETERMINISTIC, CREWAI_LLM_TYPE, OPENAI_API_KEY
from state import APP_STATE
from trading import AlpacaTrader, create_trader
from backtest import average_backtest_win_rate, run_backtest, summarize_backtest
from xau_backtest import run_xau_backtest
from xau_macd_strategy import build_xau_macd_decision, fetch_xau_m15_data, xau_decision_to_risk_decision


def _log(agent: str, message: str, level: str = "info") -> None:
    APP_STATE.add_agent_log(agent, message, level)


def _build_xau_live_decision(account) -> object | None:
    """Generate the XAU/USD 15m MACD decision and convert it into the framework's decision schema."""
    try:
        df = fetch_xau_m15_data(symbol="XAUUSD=X", lookback="7d")
        xau_decision = build_xau_macd_decision(
            df,
            account.equity,
            account.buying_power,
            risk_pct=0.015,
            symbol="XAUUSD=X",
        )
        if xau_decision and xau_decision.approved:
            APP_STATE.set_xau_indicator_values(
                price=xau_decision.entry_price,
                ema20=xau_decision.ema20,
                macd=xau_decision.macd,
                signal_line=xau_decision.macd_signal,
                histogram=xau_decision.macd_hist,
            )
            _log("Technical Market Scanner", f"XAU/USD 15m signal approved: {xau_decision.rationale}")
            return xau_decision_to_risk_decision(xau_decision)
    except Exception as exc:
        _log("Orchestrator", f"XAU 15m strategy check failed: {exc}", "warning")
    return None


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
        xau_live_decision = _build_xau_live_decision(account)
        if xau_live_decision is not None:
            approved_decisions.append(xau_live_decision)
            APP_STATE.set_active_strategy("XAU MACD 15m")
            APP_STATE.set_xau_execution_status("Signal Active")
            APP_STATE.set_xau_live_signal(xau_live_decision.rationale)
            _log("Risk Manager", xau_live_decision.rationale)
        else:
            APP_STATE.set_xau_execution_status("No Signal")
            APP_STATE.set_xau_live_signal("No XAU/USD 15m MACD trigger available in the current data window.")
            APP_STATE.set_xau_indicator_values(0.0, 0.0, 0.0, 0.0, 0.0)
        APP_STATE.set_decision_summaries(approved_decisions)

        backtest_results = run_backtest(symbols)
        win_rate = average_backtest_win_rate(backtest_results)
        APP_STATE.set_win_rate(win_rate)
        APP_STATE.set_backtest_summary(summarize_backtest(backtest_results))

        xau_result = run_xau_backtest("XAUUSD=X", lookback_days=90)
        if xau_result is not None:
            APP_STATE.set_xau_backtest_summary(xau_result.summary)
        else:
            APP_STATE.set_xau_backtest_summary("XAU/USD 15m backtest unavailable or insufficient data.")

        strategy_performance = _compute_strategy_performance(trader)
        APP_STATE.set_strategy_performance(strategy_performance)

        equity_trend = APP_STATE.get_equity_trend()
        bias = _compute_strategy_bias(win_rate, equity_trend, strategy_performance)
        APP_STATE.set_strategy_bias(bias)
        APP_STATE.set_strategy_reason(
            _build_strategy_reason(win_rate, equity_trend, strategy_performance, bias)
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
                    strategy=decision.strategy,
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


def _compute_strategy_performance(trader: AlpacaTrader) -> dict[str, float]:
    performance: dict[str, float] = {}
    try:
        positions = trader.get_positions()
        strategy_map: dict[str, str] = {}
        for entry in list(APP_STATE.trade_logs):
            if entry.strategy and entry.strategy != "Unknown":
                strategy_map[entry.symbol.upper()] = entry.strategy

        for position in positions:
            strategy = strategy_map.get(position["symbol"].upper(), "Adaptive")
            performance[strategy] = performance.get(strategy, 0.0) + float(position["unrealized_pl"])
    except Exception:
        pass
    return performance


def _compute_strategy_bias(
    win_rate: float,
    equity_trend: float,
    performance: dict[str, float],
) -> dict[str, float]:
    default = {"Momentum": 1.0, "Trend-Following": 1.0, "Defensive": 1.0}
    bias = dict(default)
    if not performance:
        return bias
    for strategy, pnl in performance.items():
        if pnl > 0:
            bias[strategy] = min(1.4, 1.0 + pnl / 1000)
        elif pnl < 0:
            bias[strategy] = max(0.7, 1.0 + pnl / 1000)
    if win_rate >= 0.55 and equity_trend > 0.01:
        bias = {k: min(1.4, v + 0.1) for k, v in bias.items()}
    elif win_rate < 0.45 or equity_trend < -0.02:
        bias = {k: max(0.7, v - 0.1) for k, v in bias.items()}
    return bias


def _build_strategy_reason(
    win_rate: float,
    equity_trend: float,
    performance: dict[str, float],
    bias: dict[str, float],
) -> str:
    reasons: list[str] = []
    if performance:
        reasons.append(
            "Real-time strategy PnL: "
            + ", ".join(f"{strategy} ${pnl:.1f}" for strategy, pnl in performance.items())
        )
    reasons.append(f"Backtest win rate {win_rate*100:.1f}%")
    reasons.append(f"Equity trend {equity_trend*100:.1f}%")
    reasons.append(
        "Bias updated: "
        + ", ".join(f"{strategy} x{weight:.2f}" for strategy, weight in bias.items())
    )
    return ". ".join(reasons)


def sync_alpaca_trade_log(trader: AlpacaTrader | None = None) -> None:
    """Refresh recent Alpaca orders into the dashboard trade log."""
    active_trader = trader or create_trader()
    if active_trader is None:
        return
    try:
        orders = active_trader.get_recent_orders(limit=20)
        positions = active_trader.get_positions()
        pnl_by_symbol = {pos["symbol"].upper(): pos["unrealized_pl"] for pos in positions}
        strategy_by_symbol = {
            entry.symbol.upper(): entry.strategy for entry in APP_STATE.trade_logs if getattr(entry, "strategy", None)
        }
        entries = []
        for order in orders:
            symbol = order["symbol"].upper()
            strategy = strategy_by_symbol.get(symbol, "Adaptive")
            pnl = pnl_by_symbol.get(symbol)
            entries.append(
                (
                    order["symbol"],
                    order["side"],
                    order["filled_qty"] or order["qty"],
                    order["status"],
                    order["id"],
                    f"{order['type']} @ avg ${order['filled_avg_price']:.2f}",
                    strategy,
                    pnl,
                )
            )
        APP_STATE.replace_trade_logs(entries)
    except Exception as exc:
        APP_STATE.set_last_error(f"Failed to sync Alpaca orders: {exc}")
