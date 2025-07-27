"""Strategy Base Model."""

from datetime import date
from typing import List
from pydantic import BaseModel

from backtester.market_data.market import MarketSnapshot
from components.job.base_model import StrategyJob
from components.trades.intent_model import TradeIntent


class Strategy(BaseModel):
    """Strategy Base Model."""

    def get_previous_price(
        self, sorted_prices: dict[date, float], target_date: date
    ) -> float | None:
        """Return the last available price before the target_date."""
        if target_date not in sorted_prices:
            return None

        all_dates = sorted(sorted_prices.keys())
        target_idx = all_dates.index(target_date)

        if target_idx == 0:
            return None  # No previous date exists

        prev_date = all_dates[target_idx - 1]
        return sorted_prices[prev_date]

    def generate_signals(
        self,
        job: StrategyJob,
        market_snapshot: MarketSnapshot,
        dates: List[date] | None = None,
    ) -> List[TradeIntent]:
        """Generate signals."""
        raise NotImplementedError(
            "Not implemented generate signals method for abstract strategy class."
        )
