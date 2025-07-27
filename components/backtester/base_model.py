"""Backtester Base Model."""

from datetime import date

from pydantic import BaseModel


class PortfolioSnapshot(BaseModel):
    date: date
    total_value: float
    cash: float
    positions: dict[str, float]
    prices: dict[str, float]


class BacktesterBaseModel(BaseModel):
    """Backtester Base Model."""
