"""XAU/USD 15-minute MACD momentum strategy logic.

This module is designed to plug into an existing bot framework that can:
- fetch 15-minute OHLCV bars for XAU/USD,
- pass account balance and buying power,
- execute market entries with attached stop loss / take profit,
- monitor open positions and close them when early-exit conditions trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from risk_manager import RiskDecision


@dataclass
class XauMacdDecision:
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int
    risk_pct: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    signal_index: int
    ema20: float
    macd: float
    macd_signal: float
    macd_hist: float
    rationale: str
    strategy: str = "XAU MACD 15m"
    approved: bool = False


def fetch_xau_m15_data(symbol: str = "XAUUSD=X", lookback: str = "10d") -> pd.DataFrame:
    """Fetch recent 15-minute bars for XAU/USD using yfinance."""
    df = yf.Ticker(symbol).history(period=lookback, interval="15m", actions=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} on 15m timeframe")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df


def compute_xau_macd_ema20(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 EMA and standard MACD values on 15-minute bars."""
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_diff"] = df["macd_hist"].diff()
    return df


def _last_macd_cross(df: pd.DataFrame) -> Optional[Tuple[int, str]]:
    """Return the index and side for the last valid MACD crossover signal."""
    if len(df) < 3:
        return None

    cross = df[[-2, -1]].copy()
    prev_macd = cross["macd"].iloc[-2]
    prev_signal = cross["macd_signal"].iloc[-2]
    current_macd = cross["macd"].iloc[-1]
    current_signal = cross["macd_signal"].iloc[-1]

    if prev_macd <= prev_signal and current_macd > current_signal:
        return len(df) - 1, "buy"
    if prev_macd >= prev_signal and current_macd < current_signal:
        return len(df) - 1, "sell"
    return None


def _risk_qty(account_balance: float, entry_price: float, stop_loss: float, risk_pct: float) -> int:
    risk_value = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0:
        return 0
    qty = int(risk_value // risk_per_unit)
    return max(qty, 0)


def build_xau_macd_decision(
    df: pd.DataFrame,
    account_balance: float,
    buying_power: float,
    risk_pct: float = 0.015,
    symbol: str = "XAUUSD=X",
) -> Optional[XauMacdDecision]:
    """Build a single XAU/USD 15m MACD decision with exact risk, stop, and take profit."""
    if df.empty:
        return None

    df = compute_xau_macd_ema20(df)
    signal = _last_macd_cross(df)
    if signal is None:
        return None

    signal_index, side = signal
    candle = df.iloc[signal_index]
    ema20 = float(candle["ema20"])
    close_price = float(candle["close"])

    if side == "buy" and close_price <= ema20:
        return None
    if side == "sell" and close_price >= ema20:
        return None

    if signal_index < 3:
        return None

    prev_three = df.iloc[signal_index - 3 : signal_index]
    if prev_three.empty or len(prev_three) < 3:
        return None

    stop_loss = float(prev_three["low"].min()) if side == "buy" else float(prev_three["high"].max())
    if side == "buy" and stop_loss >= close_price:
        return None
    if side == "sell" and stop_loss <= close_price:
        return None

    risk_per_unit = abs(close_price - stop_loss)
    if risk_per_unit <= 0:
        return None

    qty = _risk_qty(account_balance, close_price, stop_loss, risk_pct)
    if qty < 1:
        return None

    take_profit = float(close_price + risk_per_unit * 1.5) if side == "buy" else float(close_price - risk_per_unit * 1.5)
    reward_amount = abs(take_profit - close_price)
    risk_amount = abs(risk_per_unit * qty)
    rr = reward_amount / risk_per_unit if risk_per_unit else 0.0

    if qty * close_price > buying_power:
        return None

    return XauMacdDecision(
        symbol=symbol,
        side=side,
        entry_price=close_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        qty=qty,
        risk_pct=risk_pct,
        risk_amount=risk_amount,
        reward_amount=reward_amount * qty,
        risk_reward_ratio=rr,
        signal_index=signal_index,
        ema20=ema20,
        macd=float(candle["macd"]),
        macd_signal=float(candle["macd_signal"]),
        macd_hist=float(candle["macd_hist"]),
        rationale=(
            f"{side.upper()} trigger: 15m close {'above' if side == 'buy' else 'below'} 20 EMA, "
            f"MACD crossover at {close_price:.4f}. Stop loss based on prior 3-bar {'low' if side == 'buy' else 'high'} "
            f"and TP at 1.5x RR."
        ),
        approved=True,
    )


def should_exit_xau_macd_position(
    side: str,
    current_close: float,
    current_ema20: float,
    current_macd_hist: float,
    prior_macd_hist: float,
) -> Tuple[bool, Optional[str]]:
    """Return whether an open XAU/USD trade should be closed early."""
    if side == "buy":
        if current_close < current_ema20:
            return True, "Trend break: price closed below 20 EMA"
        if current_macd_hist < prior_macd_hist:
            return True, "Momentum loss: MACD histogram weakening"
    else:
        if current_close > current_ema20:
            return True, "Trend break: price closed above 20 EMA"
        if current_macd_hist > prior_macd_hist:
            return True, "Momentum loss: MACD histogram weakening"
    return False, None


def last_xau_ma20_macd_state(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the required indicator state on a 15-minute XAU/USD dataframe."""
    return compute_xau_macd_ema20(df)


def xau_decision_to_risk_decision(decision: XauMacdDecision) -> RiskDecision:
    """Adapt the XAU MACD signal into the framework's `RiskDecision` schema."""
    return RiskDecision(
        approved=decision.approved,
        symbol=decision.symbol,
        side=decision.side,
        entry_price=decision.entry_price,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        trailing_stop_pct=0.015,
        qty=decision.qty,
        risk_dollars=decision.risk_amount,
        reward_dollars=decision.reward_amount,
        risk_reward_ratio=decision.risk_reward_ratio,
        strategy=decision.strategy,
        rationale=decision.rationale,
    )


# Example integration guidance:
#
# from xau_macd_strategy import fetch_xau_m15_data, build_xau_macd_decision, should_exit_xau_macd_position
#
# df = fetch_xau_m15_data(symbol="XAUUSD=X", lookback="7d")
# decision = build_xau_macd_decision(df, account.equity, account.buying_power, risk_pct=0.015)
# if decision and decision.approved:
#     # create RiskDecision or pass decision data into broker execution.
#     trader.execute_bracket_trade(...)
#
# For live management, re-fetch the latest 15m bar each cycle and call:
# current_state = last_xau_ma20_macd_state(df)
# exit_flag, reason = should_exit_xau_macd_position(
#     side=open_side,
#     current_close=current_state["close"].iloc[-1],
#     current_ema20=current_state["ema20"].iloc[-1],
#     current_macd_hist=current_state["macd_hist"].iloc[-1],
#     prior_macd_hist=current_state["macd_hist"].iloc[-2],
# )
# if exit_flag:
#     close_market_position(symbol="XAUUSD=X")
