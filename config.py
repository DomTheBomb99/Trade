"""Application configuration and environment-backed settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Alpaca Paper Trading API — hardcoded base URL (never from env)
# Paper REST endpoint under the alpaca.markets domain
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# LLM for CrewAI agent reasoning
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CREWAI_LLM_MODEL = os.environ.get("CREWAI_LLM_MODEL", "gpt-4o-mini")
CREWAI_LLM_TYPE = os.environ.get("CREWAI_LLM_TYPE", "openai")
CREWAI_FORCE_DETERMINISTIC = os.environ.get("CREWAI_FORCE_DETERMINISTIC", "false").lower() in ("1", "true", "yes")

# Broad market universe scanned before filtering down to the final watchlist
MARKET_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "INTC", "CRM", "ORCL", "IBM", "PYPL", "SQ", "UBER", "LYFT",
    "JNJ", "PFE", "MRK", "ABBV", "CVS", "WMT", "COST", "HD",
    "LOW", "NKE", "SBUX", "DIS", "NFLX", "MCD", "JPM", "BAC",
    "GS", "MS", "V", "MA", "COIN", "GM", "F", "CAT", "DE",
    "BA", "RTX", "LMT", "UAL", "DAL", "AAL", "STZ", "KO", "PEP",
]
MAX_MARKET_SCAN_SYMBOLS = 40
MAX_STRATEGY_WATCHLIST = 18

# Risk parameters (enforced programmatically, not only by LLM)
MAX_RISK_PER_TRADE = 0.02  # 2% of equity
REWARD_TO_RISK_RATIO = 3.0  # 3:1
TRAILING_STOP_PERCENT = 0.015  # 1.5% trailing stop
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD"]

# Trading cycle interval (seconds) when auto-scan is enabled
SCAN_INTERVAL_SECONDS = 300

# In-memory log limits for dashboard display
MAX_AGENT_LOG_ENTRIES = 500
MAX_TRADE_LOG_ENTRIES = 200
