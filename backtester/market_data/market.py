"""Market Model."""

from enum import Enum
from typing import Dict, List, Optional
from backtester.market_data.base import BarData
from backtester.static_data import Symbol
import datetime
from pydantic import BaseModel


class MarketVariable(str, Enum):
    """Market Variable."""

    OPEN = "open"
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"

    MID = "mid"


class MarketSnapshot(BaseModel):
    """Market Snapshot Model."""

    dt: datetime.date
    data: Dict[Symbol, List[BarData]]

    def get(
        self,
        symbol: Symbol,
        variable: MarketVariable,
        dates: Optional[list[datetime.date] | datetime.date] = None,
        min_date: Optional[datetime.date] = None,
        max_date: Optional[datetime.date] = None,
        with_timestamps: bool = False,
    ) -> list[float] | dict[datetime.date, float]:
        """Return market variable."""
        if isinstance(dates, datetime.date):
            dates = [dates]
        bar_data = self.data[symbol]
        if not (dates and min_date and max_date):
            bar_data = [bar_datum for bar_datum in bar_data]
        if dates:
            bar_data = [
                bar_datum for bar_datum in bar_data if bar_datum.datetime in dates
            ]
        if min_date:
            bar_data = [
                bar_datum for bar_datum in bar_data if bar_datum.datetime >= min_date
            ]
        if max_date:
            bar_data = [
                bar_datum for bar_datum in bar_data if bar_datum.datetime <= max_date
            ]

        if with_timestamps:
            values = {}
        else:
            values = []

        for _bar_data in bar_data:
            match variable:
                case (
                    MarketVariable.OPEN.value
                    | MarketVariable.CLOSE.value
                    | MarketVariable.HIGH.value
                    | MarketVariable.LOW.value
                    | MarketVariable.VOLUME.value
                ):
                    value = getattr(_bar_data, variable)
                case MarketVariable.MID.value:
                    value = (
                        getattr(_bar_data, "open") + getattr(_bar_data, "close")
                    ) / 2
                case _:
                    raise NotImplementedError(f"Not implemented {variable}.")

            if with_timestamps:
                values[getattr(_bar_data, "datetime")] = value
            else:
                values.append(value)

        return values
