"""Backtest module for historical strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config import DEFAULT_WATCHLIST, REWARD_TO_RISK_RATIO, MAX_RISK_PER_TRADE
from market_data import TechnicalSnapshot, analyze_symbol, scan_watchlist
from risk_manager import RiskDecision, evaluate_trade
from sentiment import analyze_sentiment, scan_sentiment


@dataclass
class BacktestResult:
    symbol: str
    trades: int
    win_rate: float
    avg_return: float
    max_drawdown: float
    realized_pnl: float
    summary: str


def _simulate_trades(symbol: str, lookback_days: int = 180) -> BacktestResult | None:
    snapshot = analyze_symbol(symbol)
    if snapshot is None:
        return None

    history = yf.Ticker(symbol).history(period=f"{lookback_days}d", interval="1d", auto_adjust=True)
    if history.empty or len(history) < 60:
        return None

    sentiments = [analyze_sentiment(symbol)]
    account_equity = 100000.0
    buying_power = account_equity
    trades = []
    initial_price = float(history["Close"].iloc[-1])

    for idx in range(50, len(history) - 2):
        price = float(history["Close"].iloc[idx])
        price_next = float(history["Close"].iloc[idx + 1])
        technical = snapshot
        sentiment = sentiments[0]
        decision = evaluate_trade(technical, sentiment, account_equity, buying_power)
        if not decision.approved or decision.qty < 1:
            continue

        entry = price_next
        if decision.side == "buy":
            target = entry + (entry - decision.stop_loss) * REWARD_TO_RISK_RATIO
            stop = decision.stop_loss
            exit_price = target if price_next >= target else stop if price_next <= stop else price_next
        else:
            target = entry - (decision.stop_loss - entry) * REWARD_TO_RISK_RATIO
            stop = decision.stop_loss
            exit_price = target if price_next <= target else stop if price_next >= stop else price_next

        pnl = decision.qty * (exit_price - entry) if decision.side == "buy" else decision.qty * (entry - exit_price)
        trades.append(pnl)
        account_equity += pnl
        buying_power = account_equity

    if not trades:
        return BacktestResult(
            symbol=symbol,
            trades=0,
            win_rate=0.0,
            avg_return=0.0,
            max_drawdown=0.0,
            realized_pnl=0.0,
            summary="No valid simulated trades found.",
        )

    wins = sum(1 for pnl in trades if pnl > 0)
    win_rate = wins / len(trades)
    avg_return = float(np.mean(trades))
    peak = 0.0
    drawdown = 0.0
    equity_curve = np.cumsum(trades)
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)

    summary = (
        f"Simulated {len(trades)} trades for {symbol}. "
        f"Win rate {win_rate:.2%}, avg PnL ${avg_return:.2f}, max drawdown ${drawdown:.2f}."
    )

    return BacktestResult(
        symbol=symbol,
        trades=len(trades),
        win_rate=win_rate,
        avg_return=avg_return,
        max_drawdown=drawdown,
        realized_pnl=float(np.sum(trades)),
        summary=summary,
    )


def run_backtest(symbols: list[str] | None = None) -> list[BacktestResult]:
    watchlist = symbols or DEFAULT_WATCHLIST
    results: list[BacktestResult] = []
    for symbol in watchlist:
        result = _simulate_trades(symbol.strip().upper())
        if result:
            results.append(result)
    return results


def summarize_backtest(results: list[BacktestResult]) -> str:
    if not results:
        return "No backtest results available."

    lines = [result.summary for result in results]
    return "\n".join(lines)
