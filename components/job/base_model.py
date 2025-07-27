"""Strategy Job Base Model."""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from components.backtester.base_model import PortfolioSnapshot
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade


class StrategyJob(BaseModel):
    """Strategy Job Context Model."""

    strategy_name: str
    current_date: date
    tickers: List[str]

    signals: Optional[TradeIntent] = None
    equity_curve: Optional[Trade] = None
    portfolio_snapshots: Optional[PortfolioSnapshot] = None
    metrics: Optional[Dict[str, Any]] = None
