"""Run Strategies."""

from configs.example_config import MOMENTUM_CONFIG
from engine.orchestrator import Orchestrator

if __name__ == "__main__":
    for config in MOMENTUM_CONFIG:
        Orchestrator(config=config).run()
