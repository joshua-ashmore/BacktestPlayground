"""Trade Models."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from backtester.static_data import Directions, Events, Signals


class TradeSignal(BaseModel):
    """Trade Signal Model."""

    signal_type: Signals
    strength: float = Field(..., ge=0.0, le=1.0)
    trigger_price: Optional[float]
    comment: Optional[str]


class TradeEvent(BaseModel):
    """Trade Event Model."""

    timestamp: datetime
    trade_id: str
    symbol: str
    direction: Directions
    quantity: float
    price: float
    event_type: Events


class Trade(BaseModel):
    """Trade Model."""

    trade_id: str
    symbol: str
    direction: Directions
    events: List[TradeEvent]


class TradeBatch(BaseModel):
    """Trade Batch Model."""

    batch_id: str
    strategy_name: str
    trades: List[Trade]
    is_active: bool
    open_time: datetime
    close_time: Optional[datetime]
