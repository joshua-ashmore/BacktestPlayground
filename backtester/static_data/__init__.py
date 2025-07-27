"""Init."""

from enum import Enum
from typing import Annotated

Symbol = Annotated[str, "Any supported symbol."]
Currency = Annotated[str, "Any supported currency."]
Exchange = Annotated[str, "Any supported exchange."]


class AssetClasses(str, Enum):
    """Asset Class Enum."""

    EQ = "Equity"
    FX = "Forex"
    CR = "Crypto"
    FI = "Fixed Income"
    CM = "Commodities"


class Directions(str, Enum):
    """Directions Enum."""

    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"


class Signals(str, Enum):
    """Signals Enum."""

    OPEN = "open"
    CLOSE = "close"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class Events(str, Enum):
    """Events Enum."""

    ENTRY = "entry"
    SCALE = "scale"
    EXIT = "exit"


class OrderStatuses(str, Enum):
    """Order Status Enum."""

    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"


class DataSources(str, Enum):
    """Data Sources Enum."""

    YAHOO = "yahoo"
