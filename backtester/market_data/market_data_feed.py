"""Market Data Feed Models."""

import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Optional, Union, Literal, List
import yfinance as yf
import pandas as pd
import numpy as np
from backtester.market_data.sqlite_market_data_cache import SQLiteMarketCache
from backtester.static_data import DataSources, Symbol
from backtester.market_data.market import MarketSnapshot


class DataInputs(BaseModel):
    """Data Inputs Model."""

    benchmark_symbol: str
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    symbols: Optional[List[Symbol]] = None

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        if not self.end_date:
            self.end_date = datetime.datetime.now().date()
        return self


class AbstractMarketFeed(BaseModel):
    """Abstract Market Feed."""

    data_inputs: DataInputs
    source: DataSources

    market_snapshot: Optional[MarketSnapshot] = None
    dates: Optional[datetime.date] = None

    def fetch_data(self, symbols: List[str]):
        """Fetch data."""
        self.data_inputs.symbols = self.data_inputs.symbols or symbols
        self.data_inputs.symbols.append(self.data_inputs.benchmark_symbol)
        self.setup_market_data()
        cache = SQLiteMarketCache()
        if not self.market_snapshot:
            cached_data = cache.get_cached_data(
                symbols=self.data_inputs.symbols,
                start_date=self.data_inputs.start_date.strftime("%Y-%m-%d"),
                end_date=self.data_inputs.end_date.strftime("%Y-%m-%d"),
            )
            self.market_snapshot = MarketSnapshot(
                dt=self.data_inputs.end_date, **{"data": cached_data}
            )
        if not self.dates:
            self.dates = sorted(set(self.generate_dates()))

    def generate_dates(self):
        """Generate dates from market data."""
        dates = []
        for underlying in self.market_snapshot.data:
            dates.extend(
                [datum.datetime for datum in self.market_snapshot.data[underlying]]
            )
        return dates

    def setup_market_data(self):
        """
        Fetch missing data from Yahoo Finance, return full history as list of dicts.
        """
        cache = SQLiteMarketCache()
        for symbol in self.data_inputs.symbols:
            cached_data = cache.get_cached_data(
                symbols=[symbol],
                start_date=self.data_inputs.start_date.strftime("%Y-%m-%d"),
                end_date=self.data_inputs.end_date.strftime("%Y-%m-%d"),
            )
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
            missing_dates = sorted(set(all_dates.normalize()) - set(cached_dates))

            range_cached = cache.is_range_cached(
                symbol=symbol,
                start_date=self.data_inputs.start_date.strftime("%Y-%m-%d"),
                end_date=self.data_inputs.end_date.strftime("%Y-%m-%d"),
            )

            if not range_cached:
                match self.source:
                    case DataSources.YAHOO:
                        self.download_yf_data(
                            cache=cache, symbol=symbol, missing_dates=missing_dates
                        )
                    case _:
                        raise NotImplementedError(f"Not implemented {self.source}.")

    def download_yf_data(
        self,
        cache: SQLiteMarketCache,
        symbol: str,
        missing_dates: list[datetime.date],
    ):
        """Download Yahoo finance data."""
        new_start = min(missing_dates).strftime("%Y-%m-%d")
        new_end = (max(missing_dates) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_new = yf.download(tickers=symbol, start=new_start, end=new_end)
        if not df_new.empty:
            df_new.columns = [
                col[0].lower() if isinstance(col, tuple) else col.lower()
                for col in df_new.columns
            ]
            df_new = df_new.reset_index().rename(columns={"Date": "datetime"})
            df_new["datetime"] = pd.to_datetime(df_new["datetime"])
            df_new = df_new[["datetime", "open", "high", "low", "close", "volume"]]
            cache.cache_data(symbol=symbol, df=df_new)

    def download_yahoo_data(self) -> MarketSnapshot:
        """Download Yahoo finance data."""
        market_data = {}
        for symbol in self.data_inputs.symbols:
            df = yf.download(
                [symbol],
                start=self.data_inputs.start_date,
                end=self.data_inputs.end_date,
            )
            df.columns = [col[0].lower() for col in df.columns]
            df = df.reset_index().rename(columns={"Date": "datetime"})
            df["datetime"] = np.array(pd.to_datetime(df["datetime"]).dt.to_pydatetime())
            df["datetime"] = df["datetime"].astype(object)
            list_of_dicts = df.to_dict(orient="records")
            market_data[symbol] = list_of_dicts
        return MarketSnapshot(dt=self.dt, **{"data": market_data})


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
