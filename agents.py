"""CrewAI multi-agent trading team: scanner, sentiment, risk manager."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    CREWAI_AVAILABLE = False

    class Agent:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Crew:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def kickoff(self) -> None:
            raise RuntimeError("CrewAI is unavailable in this environment")

    class Process:  # type: ignore[misc]
        sequential = None

    class Task:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def tool(name: str) -> Any:  # type: ignore[return-value]
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

from config import (
    CREWAI_LLM_MODEL,
    CREWAI_LLM_TYPE,
    MARKET_UNIVERSE,
    MAX_MARKET_SCAN_SYMBOLS,
    MAX_STRATEGY_WATCHLIST,
    OPENAI_API_KEY,
)
from market_data import TechnicalSnapshot, analyze_symbol, scan_watchlist
from risk_manager import RiskDecision, evaluate_trade
from sentiment import SentimentSnapshot, analyze_sentiment, scan_sentiment
from state import APP_STATE
from trading import AccountSummary


def _build_llm() -> dict[str, Any]:
    if OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    else:
        os.environ.setdefault("OPENAI_API_KEY", "")
    return {
        "llm_type": CREWAI_LLM_TYPE,
        "model": CREWAI_LLM_MODEL,
        "temperature": 0.2,
    }


@tool("scan_technical_indicators")
def scan_technical_indicators(watchlist_csv: str) -> str:
    """Scan a comma-separated watchlist for momentum, moving averages, RSI, and MACD signals."""
    symbols = [s.strip().upper() for s in watchlist_csv.split(",") if s.strip()]
    snapshots = scan_watchlist(symbols)
    payload = [snap.to_dict() for snap in snapshots]
    return json.dumps(payload, indent=2)


@tool("analyze_single_ticker_technicals")
def analyze_single_ticker_technicals(symbol: str) -> str:
    """Run a deep technical analysis on one ticker symbol."""
    snapshot = analyze_symbol(symbol.strip().upper())
    if not snapshot:
        return json.dumps({"error": f"No market data for {symbol}"})
    return json.dumps(snapshot.to_dict(), indent=2)


@tool("scan_market_sentiment")
def scan_market_sentiment(watchlist_csv: str) -> str:
    """Scrape financial news headlines and score market sentiment for each ticker."""
    symbols = [s.strip().upper() for s in watchlist_csv.split(",") if s.strip()]
    snapshots = scan_sentiment(symbols)
    payload = [snap.to_dict() for snap in snapshots]
    return json.dumps(payload, indent=2)


@tool("analyze_single_ticker_sentiment")
def analyze_single_ticker_sentiment(symbol: str) -> str:
    """Scrape news and score sentiment for a single ticker."""
    snapshot = analyze_sentiment(symbol.strip().upper())
    return json.dumps(snapshot.to_dict(), indent=2)


@tool("scan_strategy_recommendations")
def scan_strategy_recommendations(watchlist_csv: str) -> str:
    """Generate strategy recommendations for a watchlist using technical and sentiment signals."""
    symbols = [s.strip().upper() for s in watchlist_csv.split(",") if s.strip()]
    technical_snapshots = scan_watchlist(symbols)
    recommendations: list[dict[str, Any]] = []

    for technical in technical_snapshots:
        sentiment = analyze_sentiment(technical.symbol)
        strategy = "momentum"
        if technical.signal == "SELL" or sentiment.mood == "BEARISH":
            strategy = "defensive"
        elif technical.signal == "HOLD" and sentiment.mood == "BULLISH":
            strategy = "trend_following"

        recommendations.append(
            {
                "symbol": technical.symbol,
                "technical_signal": technical.signal,
                "momentum_score": technical.momentum_score,
                "sentiment_mood": sentiment.mood,
                "recommended_strategy": strategy,
                "technical_rationale": technical.rationale,
                "sentiment_rationale": sentiment.rationale,
            }
        )

    return json.dumps(recommendations, indent=2)


@tool("evaluate_risk_parameters")
def evaluate_risk_parameters(
    technical_json: str,
    sentiment_json: str,
    account_equity: float,
    buying_power: float,
) -> str:
    """
    Critique a proposed setup and enforce 2% max risk, 3:1 reward-to-risk,
    and tight trailing stop-loss parameters. Returns approved/rejected decision.
    """
    technical_data = json.loads(technical_json)
    sentiment_data = json.loads(sentiment_json)

    technical = TechnicalSnapshot(
        symbol=technical_data["symbol"],
        price=float(technical_data["price"]),
        volume=int(technical_data["volume"]),
        avg_volume_20d=float(technical_data["avg_volume_20d"]),
        sma_20=float(technical_data["sma_20"]),
        sma_50=float(technical_data["sma_50"]),
        sma_200=float(technical_data["sma_200"]),
        rsi_14=float(technical_data["rsi_14"]),
        macd=float(technical_data["macd"]),
        macd_signal=float(technical_data["macd_signal"]),
        macd_hist=float(technical_data["macd_hist"]),
        momentum_score=float(technical_data["momentum_score"]),
        signal=technical_data["signal"],
        rationale=technical_data["rationale"],
    )
    sentiment = SentimentSnapshot(
        symbol=sentiment_data["symbol"],
        headline_count=int(sentiment_data["headline_count"]),
        bullish_count=int(sentiment_data["bullish_count"]),
        bearish_count=int(sentiment_data["bearish_count"]),
        sentiment_score=float(sentiment_data["sentiment_score"]),
        mood=sentiment_data["mood"],
        headlines=list(sentiment_data.get("headlines", [])),
        rationale=sentiment_data["rationale"],
    )

    decision = evaluate_trade(technical, sentiment, account_equity, buying_power)
    return json.dumps(decision.to_dict(), indent=2)


def build_trading_crew(watchlist: list[str], account: AccountSummary) -> Crew | None:
    if not CREWAI_AVAILABLE:
        return None

    watchlist_csv = ",".join(watchlist)
    llm = _build_llm()

    technical_scanner = Agent(
        role="Technical Market Scanner",
        goal=(
            "Continuously monitor high-volume tech stocks for momentum breakouts, "
            "moving average alignment, RSI, and MACD confirmation."
        ),
        backstory=(
            "You are a quantitative technician who reads price action objectively. "
            "You flag only high-conviction setups backed by volume and trend structure."
        ),
        tools=[
            scan_technical_indicators,
            analyze_single_ticker_technicals,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    sentiment_analyst = Agent(
        role="Web Sentiment Analyst",
        goal=(
            "Scrape financial news headers and determine whether market mood "
            "confirms or contradicts the technical scanner's findings."
        ),
        backstory=(
            "You synthesize headline sentiment from Yahoo Finance and Google News RSS. "
            "You call out narrative risk when news flow disagrees with the chart."
        ),
        tools=[
            scan_market_sentiment,
            analyze_single_ticker_sentiment,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    strategy_selector = Agent(
        role="Strategy Selector",
        goal=(
            "Choose the strongest execution strategy for each candidate based on technical, sentiment, and risk signals."
        ),
        backstory=(
            "You are a senior trader who chooses between breakout, momentum, trend-following, "
            "and defensive strategies depending on market context."
        ),
        tools=[
            scan_strategy_recommendations,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    risk_manager = Agent(
        role="Risk Manager",
        goal=(
            "Critique scanner and sentiment proposals. Enforce max 2% equity risk per trade, "
            "mandatory 3:1 reward-to-risk, and tight trailing stop-loss sizing."
        ),
        backstory=(
            "You are the capital guardian. You reject any trade that violates risk rules, "
            "even if technicals and sentiment look attractive."
        ),
        tools=[evaluate_risk_parameters],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    technical_task = Task(
        description=(
            f"Scan this watchlist: {watchlist_csv}. "
            "Use scan_technical_indicators and identify the top 1-2 tickers with "
            "strongest momentum / breakout characteristics. Summarize signals clearly."
        ),
        expected_output=(
            "JSON summary of scanned tickers with signal (BUY/SELL/HOLD), momentum score, "
            "and a shortlist of best candidates."
        ),
        agent=technical_scanner,
    )

    sentiment_task = Task(
        description=(
            f"For watchlist {watchlist_csv}, scrape headlines and score sentiment. "
            "Confirm whether news mood aligns with the technical scanner's top candidates."
        ),
        expected_output=(
            "JSON sentiment report per ticker with mood (BULLISH/BEARISH/NEUTRAL) "
            "and explicit alignment/conflict notes vs technical signals."
        ),
        agent=sentiment_analyst,
        context=[technical_task],
    )

    strategy_task = Task(
        description=(
            f"Analyze the watchlist {watchlist_csv} and recommend the best trading approach "
            "for each top candidate: breakout, momentum, trend-following, or defensive. "
            "Use technical signals and sentiment alignment to rank strategy choice."
        ),
        expected_output=(
            "JSON strategy recommendation report per ticker with chosen strategy, "
            "signal alignment, and a concise rationale."
        ),
        agent=strategy_selector,
        context=[technical_task, sentiment_task],
    )

    risk_task = Task(
        description=(
            f"Review technical and sentiment findings for {watchlist_csv}. "
            f"Account equity=${account.equity:.2f}, buying_power=${account.buying_power:.2f}. "
            "For each viable candidate, call evaluate_risk_parameters with the technical JSON, "
            "sentiment JSON, account equity, and buying power. "
            "Reject anything below 3:1 reward-to-risk or above 2% equity risk."
        ),
        expected_output=(
            "Final risk-approved trade list with entry, stop, take-profit, qty, "
            "trailing stop %, and explicit APPROVED/REJECTED verdict per ticker."
        ),
        agent=risk_manager,
        context=[technical_task, sentiment_task, strategy_task],
    )

    return Crew(
        agents=[technical_scanner, sentiment_analyst, strategy_selector, risk_manager],
        tasks=[technical_task, sentiment_task, strategy_task, risk_task],
        process=Process.sequential,
        verbose=True,
    )


def _build_broad_watchlist(watchlist: list[str]) -> list[str]:
    symbols: list[str] = []
    for symbol in watchlist + MARKET_UNIVERSE:
        candidate = symbol.strip().upper()
        if candidate and candidate not in symbols:
            symbols.append(candidate)
        if len(symbols) >= MAX_MARKET_SCAN_SYMBOLS:
            break
    return symbols


def _assign_strategy(technical: TechnicalSnapshot, sentiment: SentimentSnapshot) -> str:
    if technical.signal == "BUY" and sentiment.mood == "BULLISH":
        return "Momentum"
    if technical.signal == "BUY" and sentiment.mood == "NEUTRAL":
        return "Trend-Following"
    if technical.signal == "SELL" or sentiment.mood == "BEARISH":
        return "Defensive"
    if technical.signal == "HOLD" and sentiment.mood == "BULLISH":
        return "Trend-Following"
    return "Adaptive"


def _rank_strategies(decisions: list[RiskDecision]) -> dict[str, float]:
    strategy_counts: dict[str, int] = {}
    for decision in decisions:
        strategy_counts[decision.strategy] = strategy_counts.get(decision.strategy, 0) + 1
    total = sum(strategy_counts.values())
    return {
        strategy: round(count / total, 2) if total else 0.0
        for strategy, count in strategy_counts.items()
    }


def _strategy_weight(strategy: str) -> float:
    weight = APP_STATE.strategy_bias.get(strategy, 1.0)
    return float(weight if weight > 0 else 1.0)


def run_deterministic_pipeline(
    watchlist: list[str],
    account: AccountSummary,
    open_positions: set[str],
) -> list[RiskDecision]:
    """
    Run the same logic as the crew tools without LLM latency.
    Used for reliable execution after crew deliberation.
    """
    broad_watchlist = _build_broad_watchlist(watchlist)
    technicals = scan_watchlist(broad_watchlist)
    sentiments = {s.symbol: s for s in scan_sentiment(broad_watchlist)}

    weighted_candidates: list[RiskDecision] = []
    for technical in sorted(technicals, key=lambda t: t.momentum_score, reverse=True):
        if technical.symbol in open_positions:
            continue
        sentiment = sentiments.get(technical.symbol)
        if not sentiment:
            continue
        decision = evaluate_trade(technical, sentiment, account.equity, account.buying_power)
        if decision.approved:
            decision.strategy = _assign_strategy(technical, sentiment)
            weighted_candidates.append(decision)

    if not weighted_candidates:
        return []

    weighted_candidates.sort(
        key=lambda d: (d.risk_reward_ratio * _strategy_weight(d.strategy)),
        reverse=True,
    )
    candidates = weighted_candidates[:MAX_STRATEGY_WATCHLIST]

    strategy_rankings = _rank_strategies(candidates)
    best_strategy = max(strategy_rankings, key=strategy_rankings.get)
    APP_STATE.set_active_strategy(best_strategy)
    APP_STATE.set_strategy_rankings(strategy_rankings)
    APP_STATE.set_strategy_reason(
        f"Selected strategy based on {len(candidates)} approved candidates: "
        + ", ".join(
            f"{strategy} {pct:.0%}" for strategy, pct in strategy_rankings.items()
        )
    )
    return candidates


def extract_crew_output(crew_result: Any) -> str:
    if hasattr(crew_result, "raw"):
        return str(crew_result.raw)
    return str(crew_result)
