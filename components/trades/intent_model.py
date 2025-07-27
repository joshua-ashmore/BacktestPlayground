"""Intent Model."""

from typing import Literal
from datetime import date

from pydantic import BaseModel


class TradeIntent(BaseModel):
    """Trade Intent."""

    symbol: str
    date: date
    signal: Literal["buy", "sell", "hold"]
    weight: float
    strategy: str
