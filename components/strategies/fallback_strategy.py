"""Fallback Strategy Job."""

from datetime import date
from typing import List, Literal

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent


class FallbackStrategyJob(Strategy):
    """Fallback Strategy Job."""

    strategy_name: Literal["fallback"] = "fallback"
    symbols: List[str] = []

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> List[TradeIntent]:
        """Generate signals for all provided dates (or current_date if none given)."""
        return []
