"""Backtester Base Model."""

from pydantic import BaseModel
from datetime import date


class PortfolioSnapshot(BaseModel):
    date: date
    total_value: float
    cash: float
    positions: dict[str, float]
    prices: dict[str, float]


class BacktesterBaseModel(BaseModel):
    """Backtester Base Model."""
