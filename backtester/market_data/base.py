"""Shared Market Data Models."""

from pydantic import BaseModel, model_validator
import datetime
from typing import Annotated

Symbol = Annotated[str, "Any supported symbol."]


class Instrument(BaseModel):
    """Instrument Model."""

    symbol: Symbol
    exchange: str
    asset_class: str
    currency: str


class BarData(BaseModel):
    """Bar Data Model."""

    datetime: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @model_validator(mode="before")
    def post_validate_model(self):
        """Pre validate model."""
        self["datetime"] = self["datetime"].to_pydatetime().date()
        return self
