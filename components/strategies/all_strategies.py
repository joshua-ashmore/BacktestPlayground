"""All Strategies annotated union."""

from typing import Annotated, Union

from pydantic import Field

from components.strategies.momentum_strategy import MomentumStrategyJob


Strategies = Annotated[
    Union[MomentumStrategyJob], Field(..., discriminator="strategy_name")
]
