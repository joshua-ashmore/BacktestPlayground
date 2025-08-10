"""Market Data Feed Models."""

import datetime
from typing import Annotated, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from components.market import DataSources, Symbol
from components.market.data_sources.interactive_brokers import IBDataSource
from components.market.data_sources.yahoo import YahooDataSource
from components.market.market import MarketSnapshot
from components.market.sqlite_market_data_cache import SQLiteMarketCache


class DataInputs(BaseModel):
    """Data Inputs Model."""

    benchmark_symbol: str
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    symbols: Optional[List[Symbol]] = None

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        if not self.end_date:
            self.end_date = datetime.datetime.now().date() + datetime.timedelta(days=1)
        return self


class AbstractMarketFeed(BaseModel):
    """Abstract Market Feed."""

    data_inputs: DataInputs
    source: DataSources
    cache_data: bool = True

    market_snapshot: Optional[MarketSnapshot] = None
    dates: Optional[datetime.date] = None

    def fetch_data(self, symbols: List[str]):
        """Fetch data."""
        self.data_inputs.symbols = self.data_inputs.symbols or symbols
        self.data_inputs.symbols.append(self.data_inputs.benchmark_symbol)
        cache = SQLiteMarketCache()

        # Instantiate data source
        match self.source:
            case DataSources.YAHOO:
                data_source = YahooDataSource()
            case DataSources.INTERACTIVE_BROKERS:
                data_source = IBDataSource()

        # Cache and fetch data
        data: Dict[str, List[Dict]] = {}
        for symbol in self.data_inputs.symbols:
            if symbol not in data:
                cached_data = cache.get_cached_data(
                    symbols=[symbol],
                    start_date=self.data_inputs.start_date.strftime("%Y-%m-%d"),
                    end_date=self.data_inputs.end_date.strftime("%Y-%m-%d"),
                    source=self.source,
                )
                missing_dates = self.calculate_missing_dates(
                    symbol=symbol, cached_data=cached_data
                )
                if missing_dates:
                    fetched_data = data_source.fetch_data(
                        symbol=symbol, missing_dates=missing_dates
                    )
                    if self.cache_data and fetched_data:
                        cache.cache_data(
                            symbol=symbol,
                            df=pd.DataFrame(fetched_data),
                            source=self.source,
                        )
                    all_data = cached_data.get(symbol, []) + fetched_data
                else:
                    all_data = cached_data
                data[symbol] = all_data

        # Set market snapshot variable
        if not self.market_snapshot:
            self.market_snapshot = MarketSnapshot(
                dt=self.data_inputs.end_date, **{"data": data}
            )

        # Set dates variable
        if not self.dates:
            self.dates = sorted(set(self.generate_dates()))

    def calculate_missing_dates(self, symbol: str, cached_data: Dict):
        """Calculate missing dates from cache."""
        if cached_data != {}:
            cached_dates = sorted(
                set(
                    {
                        cached_datum["datetime"].normalize()
                        for cached_datum in cached_data[symbol]
                    }
                )
            )
        else:
            cached_dates = set()

        all_dates = pd.date_range(
            start=self.data_inputs.start_date,
            end=self.data_inputs.end_date,
            freq="B",
        )
        return sorted(set(all_dates.normalize()) - set(cached_dates))

    def generate_dates(self):
        """Generate dates from market data."""
        price_series = {}

        for symbol in self.market_snapshot.data.keys():
            closes = self.market_snapshot.get(
                symbol=symbol, variable="close", with_timestamps=True
            )
            series = pd.Series(
                data=list(closes.values()), index=list(closes.keys())
            ).sort_index()
            price_series[symbol] = series

        common_index = sorted(
            set.intersection(*(set(s.index) for s in price_series.values()))
        )
        return common_index


class RealTimeFeed(AbstractMarketFeed):
    """Real Time Feed."""

    feed_type: Literal["real-time"] = "real-time"


class HistoricalFeed(AbstractMarketFeed):
    """Historical Feed."""

    feed_type: Literal["historical"] = "historical"


class SimulatedFeed(AbstractMarketFeed):
    """Simulated Feed."""

    feed_type: Literal["simulated"] = "simulated"


MarketFeed = Annotated[
    Union[RealTimeFeed, HistoricalFeed, SimulatedFeed],
    Field(..., discriminator="feed_type"),
]
