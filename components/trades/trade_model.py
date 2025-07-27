"""Trade Model."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class Trade(BaseModel):
    """Trade Model."""

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
