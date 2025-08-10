"""All Backtester Models."""

from typing import Annotated, Union

from pydantic import Field

from components.execution.ib_engine import IBExecutionEngine
from components.execution.simple_engine import SimpleExecutionEngine

ExecutionEngines = Annotated[
    Union[SimpleExecutionEngine, IBExecutionEngine],
    Field(..., discriminator="engine_type"),
]
