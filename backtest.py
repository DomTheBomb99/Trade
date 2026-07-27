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


def _build_historical_snapshot(symbol: str, history: pd.DataFrame, index: int) -> TechnicalSnapshot | None:
    if index < 60 or index >= len(history):
        return None

    close = history["Close"].iloc[: index + 1]
    volume = history["Volume"].iloc[: index + 1]
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    rsi = _compute_rsi(close)
    macd, macd_signal, macd_hist = _compute_macd(close)

    if len(close) < 60 or np.isnan(sma_20.iloc[-1]) or np.isnan(sma_50.iloc[-1]):
        return None

    price = float(close.iloc[-1])
    latest_volume = int(volume.iloc[-1])
    avg_volume = float(volume.tail(20).mean())
    latest_rsi = float(rsi.iloc[-1])
    latest_macd = float(macd.iloc[-1])
    latest_macd_signal = float(macd_signal.iloc[-1])
    latest_macd_hist = float(macd_hist.iloc[-1])
    sma20_val = float(sma_20.iloc[-1])
    sma50_val = float(sma_50.iloc[-1])
    sma200_val = float(sma_200.iloc[-1]) if not np.isnan(sma_200.iloc[-1]) else sma50_val

    momentum_score = 0.0
    reasons: list[str] = []
    if price > sma20_val > sma50_val:
        momentum_score += 2.0
        reasons.append("Price above rising 20/50 SMA stack")
    elif price > sma20_val:
        momentum_score += 1.0
        reasons.append("Price above 20 SMA")
    if latest_macd_hist > 0 and latest_macd > latest_macd_signal:
        momentum_score += 1.5
        reasons.append("MACD bullish crossover / positive histogram")
    if 55 <= latest_rsi <= 70:
        momentum_score += 1.0
        reasons.append("RSI in bullish momentum zone (55-70)")
    elif latest_rsi > 70:
        momentum_score -= 0.5
        reasons.append("RSI overbought (>70)")
    elif latest_rsi < 30:
        momentum_score -= 1.0
        reasons.append("RSI oversold (<30)")
    if latest_volume > avg_volume * 1.2:
        momentum_score += 1.0
        reasons.append("Volume spike above 20-day average")
    pct_above_200 = ((price - sma200_val) / sma200_val) * 100 if sma200_val else 0
    if pct_above_200 > 0:
        momentum_score += 0.5
        reasons.append(f"Trading {pct_above_200:.1f}% above 200 SMA")

    if momentum_score >= 4.0:
        signal = "BUY"
    elif momentum_score <= -1.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    rationale = "; ".join(reasons) if reasons else "No strong technical conviction"
    return TechnicalSnapshot(
        symbol=symbol.upper(),
        price=price,
        volume=latest_volume,
        avg_volume_20d=avg_volume,
        sma_20=sma20_val,
        sma_50=sma50_val,
        sma_200=sma200_val,
        rsi_14=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_macd_signal,
        macd_hist=latest_macd_hist,
        momentum_score=momentum_score,
        signal=signal,
        rationale=rationale,
    )


def _simulate_trades(symbol: str, lookback_days: int = 180) -> BacktestResult | None:
    history = yf.Ticker(symbol).history(period=f"{lookback_days}d", interval="1d", auto_adjust=True)
    if history.empty or len(history) < 90:
        return None

    sentiment = analyze_sentiment(symbol)
    account_equity = 100000.0
    buying_power = account_equity
    trades: list[float] = []

    for idx in range(60, len(history) - 1):
        technical = _build_historical_snapshot(symbol, history, idx)
        if technical is None:
            continue

        decision = evaluate_trade(technical, sentiment, account_equity, buying_power)
        if not decision.approved or decision.qty < 1:
            continue

        entry = float(history["Close"].iloc[idx + 1])
        next_price = float(history["Close"].iloc[idx + 2]) if idx + 2 < len(history) else entry
        if decision.side == "buy":
            target = entry + (entry - decision.stop_loss) * REWARD_TO_RISK_RATIO
            stop = decision.stop_loss
            exit_price = target if next_price >= target else stop if next_price <= stop else next_price
            pnl = decision.qty * (exit_price - entry)
        else:
            target = entry - (decision.stop_loss - entry) * REWARD_TO_RISK_RATIO
            stop = decision.stop_loss
            exit_price = target if next_price <= target else stop if next_price >= stop else next_price
            pnl = decision.qty * (entry - exit_price)

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
    equity_curve = np.cumsum(trades)
    peak = float(np.maximum.accumulate(equity_curve).max())
    max_drawdown = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve))
    summary = (
        f"Simulated {len(trades)} trades for {symbol}. "
        f"Win rate {win_rate:.2%}, avg PnL ${avg_return:.2f}, max drawdown ${max_drawdown:.2f}."
    )

    return BacktestResult(
        symbol=symbol,
        trades=len(trades),
        win_rate=win_rate,
        avg_return=avg_return,
        max_drawdown=max_drawdown,
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


def average_backtest_win_rate(results: list[BacktestResult]) -> float:
    if not results:
        return 0.0
    return float(sum(result.win_rate for result in results) / len(results))


def summarize_backtest(results: list[BacktestResult]) -> str:
    if not results:
        return "No backtest results available."

    lines = [result.summary for result in results]
    return "\n".join(lines)
