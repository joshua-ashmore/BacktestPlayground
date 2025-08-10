"""Init."""

from datetime import date

from pydantic import BaseModel


class PortfolioSnapshot(BaseModel):
    """Portfolio Snapshot Model."""

    date: date
    total_value: float
    cash: float
    positions: dict[str, float]
    prices: dict[str, float]
