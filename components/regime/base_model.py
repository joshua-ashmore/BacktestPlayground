"""Regime Detection Model."""

from datetime import date
from typing import Dict

from pydantic import BaseModel, model_validator

from components.job.base_model import StrategyJob
from components.strategies.all_strategies import Strategies
from components.strategies.mean_reversion import PairsTradingStrategyJob
from components.strategies.momentum_strategy import MomentumStrategyJob
from components.strategies.volatility_breakout import VolatilityBreakoutStrategyJob


class AbstractRegimeEngine(BaseModel):
    """Regime Detection Engine."""

    window: int = 20
    strategy_map: Dict[str, Strategies] = {
        "trending": MomentumStrategyJob(symbols=[]),
        "volatile": VolatilityBreakoutStrategyJob(symbols=[]),
        "mean_reverting": PairsTradingStrategyJob(symbols=[]),
        # "inflation": CommodityMomentumStrategy(),
        # "defensive": QualityRotationStrategy(),
    }
    regime_history: Dict[date, str] = {}

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        return self

    def detect_regime(self, job: StrategyJob, target_date: date, previous_date: date):
        """Detect regime."""
        regime_probs = self.detect_regime_probabilities(
            job=job, previous_date=previous_date
        )
        top_regime = max(regime_probs, key=regime_probs.get)
        self.regime_history[target_date] = top_regime
        strategy = self.strategy_map[top_regime]
        job.tickers = strategy.symbols
        return strategy

    def detect_regime_probabilities(job: StrategyJob, previous_date: date) -> Dict:
        """Detect regime probabilities."""
        raise ValueError("Not implemented regime probablity method.")
