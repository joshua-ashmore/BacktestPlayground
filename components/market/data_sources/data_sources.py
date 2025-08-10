"""Data sources annotated union."""

from typing import Annotated, Union

from pydantic import Field

from components.market.data_sources.interactive_brokers import IBDataSource
from components.market.data_sources.yahoo import YahooDataSource

DataSources = Annotated[
    Union[YahooDataSource, IBDataSource], Field(..., discriminator="data_source")
]
