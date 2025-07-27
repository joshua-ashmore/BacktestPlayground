"""All Backtester Models."""

from typing import Annotated, Union

from pydantic import Field

from components.backtester.simple_backtester import SimpleBacktester

Backtesters = Annotated[
    Union[SimpleBacktester], Field(..., discriminator="backtester_type")
]
