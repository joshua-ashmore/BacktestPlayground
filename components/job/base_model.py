"""Strategy Job Base Model."""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel

from components import PortfolioSnapshot
from components.market.market import MarketSnapshot
from components.metrics.base_model import PortfolioMetrics
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade


class StrategyJob(BaseModel):
    """Strategy Job Context Model."""

    job_name: str
    current_date: date
    tickers: Optional[List[str]] = None

    signals: Optional[List[TradeIntent]] = []
    equity_curve: Optional[List[Trade]] = []
    portfolio_snapshots: Optional[List[PortfolioSnapshot]] = []
    metrics: Optional[PortfolioMetrics] = None

    market_snapshot: Optional[MarketSnapshot] = None
