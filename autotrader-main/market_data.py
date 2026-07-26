"""Technical market analysis using yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf



@dataclass
class TechnicalSnapshot:
    symbol: str
    price: float
    volume: int
    avg_volume_20d: float
    sma_20: float
    sma_50: float
    sma_200: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    momentum_score: float
    signal: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": round(self.price, 2),
            "volume": self.volume,
            "avg_volume_20d": round(self.avg_volume_20d, 0),
            "sma_20": round(self.sma_20, 2),
            "sma_50": round(self.sma_50, 2),
            "sma_200": round(self.sma_200, 2),
            "rsi_14": round(self.rsi_14, 2),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_hist": round(self.macd_hist, 4),
            "momentum_score": round(self.momentum_score, 2),
            "signal": self.signal,
            "rationale": self.rationale,
        }


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def analyze_symbol(symbol: str) -> TechnicalSnapshot | None:
    """Fetch OHLCV data and compute momentum / MA / indicator signals."""
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1y", interval="1d", auto_adjust=True)
    if history.empty or len(history) < 60:
        return None

    close = history["Close"]
    volume = history["Volume"]

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    rsi = _compute_rsi(close)
    macd, macd_signal, macd_hist = _compute_macd(close)

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


def scan_watchlist(symbols: list[str]) -> list[TechnicalSnapshot]:
    results: list[TechnicalSnapshot] = []
    for symbol in symbols:
        snapshot = analyze_symbol(symbol.strip().upper())
        if snapshot:
            results.append(snapshot)
    return results
