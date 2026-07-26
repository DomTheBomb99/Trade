"""Financial news sentiment analysis via free web sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

BULLISH_TERMS = {
    "surge",
    "rally",
    "beat",
    "upgrade",
    "breakout",
    "record",
    "growth",
    "strong",
    "bullish",
    "outperform",
    "buy",
    "soar",
    "jump",
    "gain",
    "optimistic",
    "momentum",
}

BEARISH_TERMS = {
    "fall",
    "drop",
    "miss",
    "downgrade",
    "breakdown",
    "weak",
    "bearish",
    "underperform",
    "sell",
    "plunge",
    "decline",
    "loss",
    "lawsuit",
    "investigation",
    "cut",
    "warning",
    "slump",
}


@dataclass
class SentimentSnapshot:
    symbol: str
    headline_count: int
    bullish_count: int
    bearish_count: int
    sentiment_score: float
    mood: str
    headlines: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "headline_count": self.headline_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "sentiment_score": round(self.sentiment_score, 2),
            "mood": self.mood,
            "headlines": self.headlines[:8],
            "rationale": self.rationale,
        }


def _score_headline(text: str) -> tuple[int, int]:
    lowered = text.lower()
    bullish = sum(1 for term in BULLISH_TERMS if term in lowered)
    bearish = sum(1 for term in BEARISH_TERMS if term in lowered)
    return bullish, bearish


def _fetch_yahoo_finance_headlines(symbol: str) -> list[str]:
    headlines: list[str] = []
    url = f"https://finance.yahoo.com/quote/{quote_plus(symbol)}/news"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.select("h3"):
            text = tag.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(text)
        for tag in soup.select('[data-testid="title"]'):
            text = tag.get_text(strip=True)
            if text and text not in headlines:
                headlines.append(text)
    except requests.RequestException:
        pass
    return headlines[:15]


def _fetch_google_news_rss(symbol: str) -> list[str]:
    query = quote_plus(f"{symbol} stock market")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    headlines: list[str] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:12]:
            title = entry.get("title", "").strip()
            if title:
                headlines.append(re.sub(r"\s+", " ", title))
    except Exception:
        pass
    return headlines


def analyze_sentiment(symbol: str) -> SentimentSnapshot:
    symbol = symbol.upper()
    headlines = _fetch_yahoo_finance_headlines(symbol)
    headlines.extend(_fetch_google_news_rss(symbol))

    seen: set[str] = set()
    unique_headlines: list[str] = []
    for headline in headlines:
        key = headline.lower()
        if key not in seen:
            seen.add(key)
            unique_headlines.append(headline)

    bullish_total = 0
    bearish_total = 0
    for headline in unique_headlines:
        bull, bear = _score_headline(headline)
        bullish_total += bull
        bearish_total += bear

    raw_score = bullish_total - bearish_total
    if unique_headlines:
        sentiment_score = raw_score / max(len(unique_headlines), 1)
    else:
        sentiment_score = 0.0

    if sentiment_score >= 0.35:
        mood = "BULLISH"
    elif sentiment_score <= -0.35:
        mood = "BEARISH"
    else:
        mood = "NEUTRAL"

    if not unique_headlines:
        rationale = "No recent headlines found; sentiment treated as neutral"
    else:
        rationale = (
            f"Scored {bullish_total} bullish vs {bearish_total} bearish keyword hits "
            f"across {len(unique_headlines)} headlines"
        )

    return SentimentSnapshot(
        symbol=symbol,
        headline_count=len(unique_headlines),
        bullish_count=bullish_total,
        bearish_count=bearish_total,
        sentiment_score=float(sentiment_score),
        mood=mood,
        headlines=unique_headlines,
        rationale=rationale,
    )


def scan_sentiment(symbols: list[str]) -> list[SentimentSnapshot]:
    return [analyze_sentiment(symbol) for symbol in symbols]
