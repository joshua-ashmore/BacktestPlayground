"""Yahoo Finance Data Source."""

import datetime
from typing import Dict, List, Literal

import pandas as pd
import yfinance as yf
from pydantic import BaseModel


class YahooDataSource(BaseModel):
    """Yahoo Finance Data Source."""

    data_source: Literal["yahoo"] = "yahoo"

    def fetch_data(
        self,
        symbol: str,
        missing_dates: list[datetime.date],
    ) -> List[Dict]:
        """Download Yahoo finance data."""
        new_start = min(missing_dates).strftime("%Y-%m-%d")
        new_end = (max(missing_dates) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        data_frame = yf.download(tickers=symbol, start=new_start, end=new_end)
        if not data_frame.empty:
            data_frame.columns = [
                col[0].lower() if isinstance(col, tuple) else col.lower()
                for col in data_frame.columns
            ]
            data_frame = data_frame.reset_index().rename(columns={"Date": "datetime"})
            data_frame["datetime"] = pd.to_datetime(data_frame["datetime"])
            data_frame = data_frame[
                ["datetime", "open", "high", "low", "close", "volume"]
            ]
        return data_frame.to_dict("records")
