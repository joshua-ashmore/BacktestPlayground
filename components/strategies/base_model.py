"""Strategy Base Model."""

from datetime import date
from typing import List

from pydantic import BaseModel

from components.job.base_model import StrategyJob
from components.market.market import MarketSnapshot
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade


class Strategy(BaseModel):
    """Strategy Base Model."""

    symbols: List[str]

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
        dates = dates[1:] or [job.current_date]
        trade_intents: List[TradeIntent] = []
        for target_date in dates:
            trade_intents.extend(
                self.generate_signal_on_date(
                    job=job, market_snapshot=market_snapshot, target_date=target_date
                )
            )
        return trade_intents

    def generate_signal_on_date(
        self,
        job: StrategyJob,
        market_snapshot: MarketSnapshot,
        target_date: date,
    ) -> List[TradeIntent]:
        """Generate signal on given date."""
        raise NotImplementedError(
            "Not implemented generate signals method for abstract strategy class."
        )

    def generate_exit_signals(
        self, job: StrategyJob, trade: Trade, current_date: date
    ) -> bool:
        """Generate exit signals."""
        return False

    def get_required_lookback_days(self) -> int:
        """The longest historical window we need for calculations."""
        raise NotImplementedError("Not implemented required lookback days.")
