"""Trading orchestration: CrewAI deliberation + Alpaca execution."""

from __future__ import annotations

import traceback

from config import DEFAULT_WATCHLIST
from state import APP_STATE
from trading import AlpacaTrader, create_trader
from backtest import run_backtest, summarize_backtest


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
        _log(
            "Orchestrator",
            f"Paper account equity ${account.equity:,.2f} | buying power ${account.buying_power:,.2f}",
        )

        market_open = trader.is_market_open()
        if not market_open:
            msg = "Market is closed. Analysis completed; orders skipped until market open."
            _log("Orchestrator", msg, "warning")
        else:
            _log("Orchestrator", "Market is open. Trade execution enabled.")

        _log("Technical Market Scanner", "Scanning watchlist for momentum, MAs, RSI, MACD...")
        _log("Web Sentiment Analyst", "Scraping Yahoo Finance + Google News headlines...")

        crew = build_trading_crew(symbols, account)
        if crew is not None and CREWAI_AVAILABLE:
            _log("Orchestrator", "CrewAI team collaborating on trade decision...")
            try:
                crew_result = crew.kickoff()
                crew_summary = extract_crew_output(crew_result)
                APP_STATE.set_last_scan_summary(crew_summary[:4000])
                _log("Orchestrator", "Crew deliberation complete. Parsing risk-approved setups...")
                _log("Risk Manager", crew_summary[:500] + ("..." if len(crew_summary) > 500 else ""))
            except Exception as crew_error:
                _log(
                    "Orchestrator",
                    f"CrewAI deliberation failed ({crew_error}). Falling back to deterministic pipeline.",
                    "warning",
                )
                APP_STATE.set_last_scan_summary(f"Crew fallback: {crew_error}")
        else:
            _log(
                "Orchestrator",
                "CrewAI is unavailable. Using deterministic strategy pipeline only.",
                "warning",
            )
            APP_STATE.set_last_scan_summary("CrewAI unavailable; deterministic pipeline only.")

        open_positions = trader.get_open_position_symbols()
        approved_decisions = run_deterministic_pipeline(symbols, account, open_positions)

        backtest_results = run_backtest(symbols)
        APP_STATE.set_backtest_summary(summarize_backtest(backtest_results))

        if not approved_decisions:
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
