"""Thread-safe shared state for agent logs and trade history."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import MAX_AGENT_LOG_ENTRIES, MAX_TRADE_LOG_ENTRIES
from risk_manager import RiskDecision


@dataclass
class AgentLogEntry:
    timestamp: str
    agent: str
    message: str
    level: str = "info"


@dataclass
class TradeLogEntry:
    timestamp: str
    symbol: str
    side: str
    qty: float
    status: str
    order_id: str
    details: str = ""


@dataclass
class DecisionSummaryEntry:
    symbol: str
    side: str
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    risk_reward_ratio: float
    strategy: str
    rationale: str

    @classmethod
    def from_decision(cls, decision: RiskDecision) -> "DecisionSummaryEntry":
        return cls(
            symbol=decision.symbol,
            side=decision.side,
            qty=decision.qty,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            trailing_stop_pct=decision.trailing_stop_pct,
            risk_reward_ratio=decision.risk_reward_ratio,
            strategy=decision.strategy,
            rationale=decision.rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "trailing_stop_pct": round(self.trailing_stop_pct, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "strategy": self.strategy,
            "rationale": self.rationale,
        }


@dataclass
class TradingState:
    agent_logs: deque = field(default_factory=lambda: deque(maxlen=MAX_AGENT_LOG_ENTRIES))
    trade_logs: deque = field(default_factory=lambda: deque(maxlen=MAX_TRADE_LOG_ENTRIES))
    watchlist: list[str] = field(default_factory=list)
    active_mode: str = "deterministic"
    active_strategy: str = "momentum"
    strategy_reason: str = ""
    win_rate: float = 0.0
    strategy_rankings: dict[str, float] = field(default_factory=dict)
    last_scan_summary: str = ""
    backtest_summary: str = ""
    decision_summaries: list[DecisionSummaryEntry] = field(default_factory=list)
    is_scanning: bool = False
    last_error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_agent_log(self, agent: str, message: str, level: str = "info") -> None:
        entry = AgentLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            message=message,
            level=level,
        )
        with self._lock:
            self.agent_logs.appendleft(entry)

    def add_trade_log(
        self,
        symbol: str,
        side: str,
        qty: float,
        status: str,
        order_id: str,
        details: str = "",
    ) -> None:
        entry = TradeLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side=side,
            qty=qty,
            status=status,
            order_id=order_id,
            details=details,
        )
        with self._lock:
            self.trade_logs.appendleft(entry)

    def set_watchlist(self, symbols: list[str]) -> None:
        with self._lock:
            self.watchlist = symbols

    def get_watchlist(self) -> list[str]:
        with self._lock:
            return list(self.watchlist)

    def set_scanning(self, value: bool) -> None:
        with self._lock:
            self.is_scanning = value

    def set_last_scan_summary(self, summary: str) -> None:
        with self._lock:
            self.last_scan_summary = summary

    def set_backtest_summary(self, summary: str) -> None:
        with self._lock:
            self.backtest_summary = summary

    def set_active_mode(self, mode: str) -> None:
        with self._lock:
            self.active_mode = mode

    def set_active_strategy(self, strategy: str) -> None:
        with self._lock:
            self.active_strategy = strategy

    def set_strategy_reason(self, reason: str) -> None:
        with self._lock:
            self.strategy_reason = reason

    def set_win_rate(self, rate: float) -> None:
        with self._lock:
            self.win_rate = rate

    def set_strategy_rankings(self, rankings: dict[str, float]) -> None:
        with self._lock:
            self.strategy_rankings = rankings

    def set_decision_summaries(self, decisions: list[DecisionSummaryEntry] | list[RiskDecision]) -> None:
        with self._lock:
            if decisions and isinstance(decisions[0], RiskDecision):
                self.decision_summaries = [DecisionSummaryEntry.from_decision(decision) for decision in decisions]  # type: ignore[arg-type]
            else:
                self.decision_summaries = decisions  # type: ignore[assignment]

    def replace_trade_logs(
        self,
        entries: list[tuple[str, str, float, str, str, str]],
    ) -> None:
        """Replace trade logs with fresh Alpaca sync data."""
        with self._lock:
            self.trade_logs.clear()
            for symbol, side, qty, status, order_id, details in entries:
                self.trade_logs.appendleft(
                    TradeLogEntry(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        status=status,
                        order_id=order_id,
                        details=details,
                    )
                )

    def set_last_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "agent_logs": list(self.agent_logs),
                "trade_logs": list(self.trade_logs),
                "watchlist": list(self.watchlist),
                "active_mode": self.active_mode,
                "active_strategy": self.active_strategy,
                "strategy_reason": self.strategy_reason,
                "win_rate": self.win_rate,
                "strategy_rankings": dict(self.strategy_rankings),
                "last_scan_summary": self.last_scan_summary,
                "backtest_summary": self.backtest_summary,
                "decision_summaries": [entry.to_dict() for entry in self.decision_summaries],
                "is_scanning": self.is_scanning,
                "last_error": self.last_error,
            }


# Global singleton used by orchestrator and Streamlit UI
APP_STATE = TradingState()
