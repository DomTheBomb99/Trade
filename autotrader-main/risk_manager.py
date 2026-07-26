"""Deterministic risk management: position sizing, R:R, trailing stops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import MAX_RISK_PER_TRADE, REWARD_TO_RISK_RATIO, TRAILING_STOP_PERCENT
from market_data import TechnicalSnapshot
from sentiment import SentimentSnapshot


@dataclass
class RiskDecision:
    approved: bool
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    qty: int
    risk_dollars: float
    reward_dollars: float
    risk_reward_ratio: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "trailing_stop_pct": round(self.trailing_stop_pct, 4),
            "qty": self.qty,
            "risk_dollars": round(self.risk_dollars, 2),
            "reward_dollars": round(self.reward_dollars, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "rationale": self.rationale,
        }


def _atr_proxy(price: float, technical: TechnicalSnapshot) -> float:
    """Use SMA distance as a volatility proxy when ATR is unavailable."""
    distance = abs(price - technical.sma_20) / price
    base = max(distance, TRAILING_STOP_PERCENT)
    return min(base, 0.04)


def evaluate_trade(
    technical: TechnicalSnapshot,
    sentiment: SentimentSnapshot,
    account_equity: float,
    buying_power: float,
) -> RiskDecision:
    symbol = technical.symbol
    price = technical.price

    if account_equity <= 0:
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side="hold",
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            trailing_stop_pct=TRAILING_STOP_PERCENT,
            qty=0,
            risk_dollars=0,
            reward_dollars=0,
            risk_reward_ratio=0,
            rationale="Account equity unavailable or zero",
        )

    side = "hold"
    if technical.signal == "BUY" and sentiment.mood in {"BULLISH", "NEUTRAL"}:
        side = "buy"
    elif technical.signal == "SELL" and sentiment.mood in {"BEARISH", "NEUTRAL"}:
        side = "sell"
    elif technical.signal == "BUY" and sentiment.mood == "BEARISH":
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side="hold",
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            trailing_stop_pct=TRAILING_STOP_PERCENT,
            qty=0,
            risk_dollars=0,
            reward_dollars=0,
            risk_reward_ratio=0,
            rationale="Rejected: technical BUY conflicts with bearish sentiment",
        )
    else:
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side="hold",
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            trailing_stop_pct=TRAILING_STOP_PERCENT,
            qty=0,
            risk_dollars=0,
            reward_dollars=0,
            risk_reward_ratio=0,
            rationale="No aligned technical + sentiment setup for entry",
        )

    stop_distance_pct = max(_atr_proxy(price, technical), TRAILING_STOP_PERCENT)
    stop_distance = price * stop_distance_pct
    reward_distance = stop_distance * REWARD_TO_RISK_RATIO

    if side == "buy":
        stop_loss = price - stop_distance
        take_profit = price + reward_distance
    else:
        stop_loss = price + stop_distance
        take_profit = price - reward_distance

    max_risk_dollars = account_equity * MAX_RISK_PER_TRADE
    qty = int(max_risk_dollars // stop_distance) if stop_distance > 0 else 0

    if qty < 1:
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=stop_distance_pct,
            qty=0,
            risk_dollars=0,
            reward_dollars=0,
            risk_reward_ratio=REWARD_TO_RISK_RATIO,
            rationale="Position size below 1 share after 2% risk cap",
        )

    notional = qty * price
    if notional > buying_power:
        qty = int(buying_power // price)
        if qty < 1:
            return RiskDecision(
                approved=False,
                symbol=symbol,
                side=side,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=stop_distance_pct,
                qty=0,
                risk_dollars=0,
                reward_dollars=0,
                risk_reward_ratio=REWARD_TO_RISK_RATIO,
                rationale="Insufficient buying power for minimum position",
            )

    risk_dollars = qty * stop_distance
    reward_dollars = qty * reward_distance
    actual_rr = reward_dollars / risk_dollars if risk_dollars > 0 else 0

    if actual_rr < REWARD_TO_RISK_RATIO:
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=stop_distance_pct,
            qty=qty,
            risk_dollars=risk_dollars,
            reward_dollars=reward_dollars,
            risk_reward_ratio=actual_rr,
            rationale=f"Rejected: reward-to-risk {actual_rr:.2f} below required {REWARD_TO_RISK_RATIO}:1",
        )

    risk_pct = (risk_dollars / account_equity) * 100
    if risk_pct > MAX_RISK_PER_TRADE * 100 + 0.01:
        return RiskDecision(
            approved=False,
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=stop_distance_pct,
            qty=qty,
            risk_dollars=risk_dollars,
            reward_dollars=reward_dollars,
            risk_reward_ratio=actual_rr,
            rationale=f"Rejected: risk {risk_pct:.2f}% exceeds 2% equity cap",
        )

    return RiskDecision(
        approved=True,
        symbol=symbol,
        side=side,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop_pct=stop_distance_pct,
        qty=qty,
        risk_dollars=risk_dollars,
        reward_dollars=reward_dollars,
        risk_reward_ratio=actual_rr,
        rationale=(
            f"Approved {side.upper()} {qty} shares at ${price:.2f}; "
            f"risk ${risk_dollars:.2f} ({risk_pct:.2f}% equity), "
            f"target ${take_profit:.2f}, stop ${stop_loss:.2f}, "
            f"R:R {actual_rr:.1f}:1, trailing stop {stop_distance_pct*100:.2f}%"
        ),
    )
