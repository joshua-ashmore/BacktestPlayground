"""All Strategies annotated union."""

from typing import Annotated, Union

from pydantic import Field

from components.strategies.gradient_momentum import GradientMomentumStrategy
from components.strategies.fallback_strategy import FallbackStrategyJob
from components.strategies.mean_reversion import PairsTradingStrategyJob
from components.strategies.momentum_strategy import MomentumStrategyJob
from components.strategies.volatility_breakout import VolatilityBreakoutStrategyJob

Strategies = Annotated[
    Union[
        MomentumStrategyJob,
        PairsTradingStrategyJob,
        VolatilityBreakoutStrategyJob,
        GradientMomentumStrategy,
        FallbackStrategyJob,
    ],
    Field(..., discriminator="strategy_name"),
]
