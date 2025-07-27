"""Strategy Models."""

from typing import List, Optional

from pydantic import BaseModel

from backtester.market_data.base import Instrument
from backtester.market_data.market import Symbol
from backtester.static_data import Directions
from backtester.trade.trade import TradeSignal


class AbstractStrategyConfig(BaseModel):
    """Strategy Config Model."""

    instruments: List[Instrument]

    def generate_parameters(self):
        """Generate parameters - abstract method."""
        raise NotImplementedError(
            "Not implemented generate_parameters method for abstract strategy."
        )

    def evaluate(self):
        """Evaluate strategy - abstract method."""
        raise NotImplementedError(
            "Not implemented evaluate method for abstract strategy."
        )


class StrategyState(BaseModel):
    """Strategy State Model."""

    strategy_name: str
    current_positions: dict
    open_batches: List[str]
    custom_state: dict = {}  # signal model state


class TradeIntent(BaseModel):
    """Trade Intent Model."""

    symbol: Symbol
    direction: Directions
    notional_pct: float
    target_price: Optional[float] = None
    signal: TradeSignal
