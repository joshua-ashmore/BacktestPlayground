"""Intent Model."""

from datetime import date
from typing import Dict, List, Literal

from pydantic import BaseModel


class TradeIntentLeg(BaseModel):
    """Trade Intent Leg."""

    symbol: str
    date: date
    signal: Literal["buy", "sell"]
    weight: float
    strategy: str


class TradeIntent(BaseModel):
    """Trade Intent."""

    legs: List[TradeIntentLeg]
    date: date
    metadata: Dict[str, float | int | List[float]] = {}
