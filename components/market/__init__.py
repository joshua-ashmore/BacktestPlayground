"""Init."""

from enum import Enum
from typing import Annotated

Symbol = Annotated[str, "Any supported symbol."]


class Directions(str, Enum):
    """Directions Enum."""

    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"


class DataSources(str, Enum):
    """Data Sources Enum."""

    YAHOO = "yahoo"
    INTERACTIVE_BROKERS = "ib"
