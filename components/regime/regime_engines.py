"""Regime Engines."""

from typing import Annotated, Union

from pydantic import Field

from components.regime.hmm_engine import HMMRegimeEngine
from components.regime.multi_hmm_engine import MultiHMMEngine
from components.regime.naive_engine import NaiveRegimeEngine

RegimeEngines = Annotated[
    Union[NaiveRegimeEngine, HMMRegimeEngine, MultiHMMEngine],
    Field(..., discriminator="engine_type"),
]
