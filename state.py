"""Thread-safe shared state for agent logs and trade history."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import MAX_AGENT_LOG_ENTRIES, MAX_TRADE_LOG_ENTRIES


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
class TradingState:
    agent_logs: deque = field(default_factory=lambda: deque(maxlen=MAX_AGENT_LOG_ENTRIES))
    trade_logs: deque = field(default_factory=lambda: deque(maxlen=MAX_TRADE_LOG_ENTRIES))
    watchlist: list[str] = field(default_factory=list)
    last_scan_summary: str = ""
    backtest_summary: str = ""
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
                "last_scan_summary": self.last_scan_summary,
                "backtest_summary": self.backtest_summary,
                "is_scanning": self.is_scanning,
                "last_error": self.last_error,
            }


# Global singleton used by orchestrator and Streamlit UI
APP_STATE = TradingState()
