"""CrewAI multi-agent trading team: scanner, sentiment, risk manager."""

from __future__ import annotations

import json
import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_openai import ChatOpenAI

from config import CREWAI_LLM_MODEL, OPENAI_API_KEY
from market_data import TechnicalSnapshot, analyze_symbol, scan_watchlist
from risk_manager import RiskDecision, evaluate_trade
from sentiment import SentimentSnapshot, analyze_sentiment, scan_sentiment
from trading import AccountSummary


def _build_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", "")
    return ChatOpenAI(model=CREWAI_LLM_MODEL, temperature=0.2)


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


def build_trading_crew(watchlist: list[str], account: AccountSummary) -> Crew:
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
        context=[technical_task, sentiment_task],
    )

    return Crew(
        agents=[technical_scanner, sentiment_analyst, risk_manager],
        tasks=[technical_task, sentiment_task, risk_task],
        process=Process.sequential,
        verbose=True,
    )


def run_deterministic_pipeline(
    watchlist: list[str],
    account: AccountSummary,
    open_positions: set[str],
) -> list[RiskDecision]:
    """
    Run the same logic as the crew tools without LLM latency.
    Used for reliable execution after crew deliberation.
    """
    technicals = scan_watchlist(watchlist)
    sentiments = {s.symbol: s for s in scan_sentiment(watchlist)}

    candidates: list[RiskDecision] = []
    for technical in sorted(technicals, key=lambda t: t.momentum_score, reverse=True):
        if technical.symbol in open_positions:
            continue
        sentiment = sentiments.get(technical.symbol)
        if not sentiment:
            continue
        decision = evaluate_trade(technical, sentiment, account.equity, account.buying_power)
        if decision.approved:
            candidates.append(decision)
            if len(candidates) >= 2:
                break
    return candidates


def extract_crew_output(crew_result: Any) -> str:
    if hasattr(crew_result, "raw"):
        return str(crew_result.raw)
    return str(crew_result)
