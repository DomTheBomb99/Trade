"""Alpaca Paper Trading execution layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import alpaca_trade_api as tradeapi

from config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY
from risk_manager import RiskDecision


@dataclass
class AccountSummary:
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float


class AlpacaTrader:
    """Wrapper around alpaca-trade-api for paper trading."""

    def __init__(self) -> None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
            api_version="v2",
        )

    def get_account(self) -> AccountSummary:
        account = self.api.get_account()
        return AccountSummary(
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
        )

    def get_open_position_symbols(self) -> set[str]:
        positions = self.api.list_positions()
        return {position.symbol.upper() for position in positions}

    def execute_bracket_trade(self, decision: RiskDecision) -> dict[str, Any]:
        if not decision.approved or decision.qty < 1:
            return {"executed": False, "reason": decision.rationale}

        side = "buy" if decision.side == "buy" else "sell"
        trailing_percent = round(decision.trailing_stop_pct * 100, 2)

        order = self.api.submit_order(
            symbol=decision.symbol,
            qty=decision.qty,
            side=side,
            type="market",
            time_in_force="gtc",
            order_class="bracket",
            take_profit={"limit_price": round(decision.take_profit, 2)},
            stop_loss={
                "stop_price": round(decision.stop_loss, 2),
                "limit_price": round(decision.stop_loss * (0.995 if side == "buy" else 1.005), 2),
            },
        )

        return {
            "executed": True,
            "symbol": decision.symbol,
            "side": side,
            "qty": decision.qty,
            "entry_order_id": str(order.id),
            "take_profit": decision.take_profit,
            "stop_loss": decision.stop_loss,
            "trailing_stop_pct": trailing_percent,
        }

    def get_recent_orders(self, limit: int = 25) -> list[dict[str, Any]]:
        orders = self.api.list_orders(status="all", limit=limit, nested=True)
        results: list[dict[str, Any]] = []
        for order in orders:
            results.append(
                {
                    "id": str(order.id),
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty) if order.qty else 0,
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "type": order.type,
                    "status": order.status,
                    "submitted_at": str(order.submitted_at),
                    "filled_avg_price": float(order.filled_avg_price or 0),
                }
            )
        return results

    def is_market_open(self) -> bool:
        clock = self.api.get_clock()
        return bool(clock.is_open)


def create_trader() -> AlpacaTrader | None:
    try:
        return AlpacaTrader()
    except ValueError:
        return None
