"""Backtesting support for XAU/USD 15m MACD strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from xau_macd_strategy import compute_xau_macd_ema20


@dataclass
class XauBacktestResult:
    symbol: str
    trades: int
    win_rate: float
    avg_return: float
    total_pnl: float
    max_drawdown: float
    summary: str
    trades_detail: list[dict[str, Any]]


def _build_historical_state(symbol: str, history: pd.DataFrame, index: int) -> pd.DataFrame | None:
    if index < 26 or index >= len(history):
        return None
    data = history.iloc[: index + 1].copy()
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    state = compute_xau_macd_ema20(data)
    if state["ema20"].isna().iloc[-1]:
        return None
    return state


def _simulate_xau_trades(symbol: str, lookback_days: int = 90) -> XauBacktestResult | None:
    history = yf.Ticker(symbol).history(period=f"{lookback_days}d", interval="15m", actions=False)
    if history.empty or len(history) < 100:
        return None

    history = history.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    trades: list[dict[str, Any]] = []
    balance = 100000.0

    for idx in range(27, len(history) - 1):
        state = _build_historical_state(symbol, history, idx)
        if state is None:
            continue

        row = state.iloc[-1]
        prev_row = state.iloc[-2]

        if prev_row["macd"] <= prev_row["macd_signal"] and row["macd"] > row["macd_signal"] and row["close"] > row["ema20"]:
            side = "buy"
        elif prev_row["macd"] >= prev_row["macd_signal"] and row["macd"] < row["macd_signal"] and row["close"] < row["ema20"]:
            side = "sell"
        else:
            continue

        prev_three = state.iloc[-4:-1]
        if len(prev_three) < 3:
            continue
        stop_loss = float(prev_three["low"].min()) if side == "buy" else float(prev_three["high"].max())
        entry_price = float(row["close"])
        risk_amount = abs(entry_price - stop_loss)
        if risk_amount == 0 or (side == "buy" and stop_loss >= entry_price) or (side == "sell" and stop_loss <= entry_price):
            continue

        qty = int((balance * 0.015) // risk_amount)
        if qty < 1:
            continue

        take_profit = float(entry_price + risk_amount * 1.5) if side == "buy" else float(entry_price - risk_amount * 1.5)
        stop_price = stop_loss
        exit_price = None
        exit_type = None
        pnl = 0.0

        for j in range(idx + 1, len(state)):
            current = state.iloc[j]
            if side == "buy":
                if current["low"] <= stop_price:
                    exit_price = stop_price
                    exit_type = "stop"
                    break
                if current["high"] >= take_profit:
                    exit_price = take_profit
                    exit_type = "tp"
                    break
                if current["close"] < current["ema20"]:
                    exit_price = float(current["close"])
                    exit_type = "trend_break"
                    break
                prior_hist = state["macd_hist"].iloc[j - 1]
                if current["macd_hist"] < prior_hist:
                    exit_price = float(current["close"])
                    exit_type = "momentum_loss"
                    break
            else:
                if current["high"] >= stop_price:
                    exit_price = stop_price
                    exit_type = "stop"
                    break
                if current["low"] <= take_profit:
                    exit_price = take_profit
                    exit_type = "tp"
                    break
                if current["close"] > current["ema20"]:
                    exit_price = float(current["close"])
                    exit_type = "trend_break"
                    break
                prior_hist = state["macd_hist"].iloc[j - 1]
                if current["macd_hist"] > prior_hist:
                    exit_price = float(current["close"])
                    exit_type = "momentum_loss"
                    break

        if exit_price is None:
            final = state.iloc[-1]
            exit_price = float(final["close"])
            exit_type = "hold"

        pnl = qty * ((exit_price - entry_price) if side == "buy" else (entry_price - exit_price))
        balance += pnl
        trades.append(
            {
                "entry_index": idx,
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_price,
                "take_profit": take_profit,
                "exit_price": exit_price,
                "exit_type": exit_type,
                "pnl": pnl,
            }
        )

    if not trades:
        return XauBacktestResult(
            symbol=symbol,
            trades=0,
            win_rate=0.0,
            avg_return=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            summary="No XAU/USD MACD signals were found in the backtest window.",
            trades_detail=[],
        )

    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    win_rate = float(wins) / len(trades)
    avg_return = float(np.mean([trade["pnl"] for trade in trades]))
    equity_curve = np.cumsum([trade["pnl"] for trade in trades])
    peak = float(np.maximum.accumulate(equity_curve).max())
    max_dd = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve))

    total_pnl = float(np.sum([trade["pnl"] for trade in trades]))
    return XauBacktestResult(
        symbol=symbol,
        trades=len(trades),
        win_rate=win_rate,
        avg_return=avg_return,
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        summary=(
            f"XAU/USD 15m MACD backtest: {len(trades)} trades, win rate {win_rate:.1%}, "
            f"avg PnL ${avg_return:.2f}, total PnL ${total_pnl:.2f}, max DD ${max_dd:.2f}."
        ),
        trades_detail=trades,
    )


def run_xau_backtest(symbol: str = "XAUUSD=X", lookback_days: int = 90) -> XauBacktestResult | None:
    return _simulate_xau_trades(symbol, lookback_days)
