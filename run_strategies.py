"""Run Strategies."""

from engine.orchestrator import Orchestrator
from configs.example_config import MOMENTUM_CONFIG


if __name__ == "main":
    for config in MOMENTUM_CONFIG:
        Orchestrator(config=config).run()
