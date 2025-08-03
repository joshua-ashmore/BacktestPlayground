"""Trade Model."""

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator


class TradeLeg(BaseModel):
    """Trade Leg Model."""

    symbol: str
    open_date: date
    close_date: Optional[date] = None
    quantity: float
    price: float
    close_price: Optional[float] = None
    notional: float
    strategy: str
    side: Literal["buy", "sell"]
    pnl: Optional[float] = None
    direction_multiplier: Optional[float] = 0

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.direction_multiplier = 1 if self.side == "buy" else -1
        return self


class Trade(BaseModel):
    """Trade Model."""

    legs: List[TradeLeg]
    metadata: Dict[str, float | int | List[float]] = {}
