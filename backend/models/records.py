"""Typed Pydantic Rows."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PriceRecord(BaseModel):
    """SQLite Price Record Model."""

    ticker: str
    date: str
    close: float


# TODO: change this to be a unioned model of all future strategies
class StrategyConfig(BaseModel):
    """SQLite Strategy Config Model."""

    lookback: Optional[int] = None
    entry_z: Optional[float] = None
    exit_z: Optional[float] = None
    symbol: Optional[str] = None
    other: Optional[dict] = None


class StrategyRun(BaseModel):
    """SQLite Strategy Run Model."""

    id: Optional[int] = None
    name: str
    config: StrategyConfig
    timestamp: Optional[datetime] = None


class DailyReturn(BaseModel):
    """SQLite Daily Return Model."""

    run_id: int
    date: str
    symbol: str
    pnl: float


class StrategyMetrics(BaseModel):
    """SQLite Strategy Metrics Model."""

    run_id: int
    sharpe: float
    total_return: float
    max_drawdown: float
