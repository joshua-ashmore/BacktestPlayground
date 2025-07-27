"""Orchestrator."""

from typing import (
    Callable,
    List,
)
import pandas as pd
from pydantic import BaseModel

from backtester.market_data.market_data_feed import (
    MarketFeed,
)
from components.backtester.all_backtesters import Backtesters
from components.job.base_model import StrategyJob
from components.metrics.base_model import MetricsEngine
from components.reporter.base_model import PDFReporter
from components.strategies.all_strategies import Strategies


class DBInterface(BaseModel):
    def save(self, table: str, data: pd.DataFrame, job: "StrategyJob") -> None:
        raise NotImplementedError


class Hook(BaseModel):
    def before(self, stage: str, job: "StrategyJob"):
        raise NotImplementedError

    def after(self, stage: str, job: "StrategyJob"):
        raise NotImplementedError


class OrchestratorConfig(BaseModel):
    """Orchestrator Config."""

    job: StrategyJob
    market_feed: MarketFeed
    strategy: Strategies
    backtester: Backtesters
    metrics_engine: MetricsEngine
    reporter: PDFReporter
    hooks: List[Hook] = []


class Orchestrator(BaseModel):
    """Orchestrator."""

    config: OrchestratorConfig

    def _run_phase(
        self, fn: Callable[[StrategyJob], None], stage: str, job: StrategyJob
    ):
        for hook in self.config.hooks:
            hook.before(stage=stage, job=job)
        fn(job)
        for hook in self.config.hooks:
            hook.after(stage=stage, job=job)

    def run(self):
        """Run strategy."""
        self._run_phase(fn=self._load_data, stage="load_data", job=self.config.job)
        self._run_phase(
            fn=self._generate_signals, stage="generate_signals", job=self.config.job
        )
        self._run_phase(
            fn=self._execute_trades, stage="execute_trades", job=self.config.job
        )
        self._run_phase(fn=self._run_backtest, stage="backtest", job=self.config.job)
        self._run_phase(fn=self._compute_metrics, stage="metrics", job=self.config.job)
        # self._run_phase(fn=self._save_results, stage="save", job=self.config.job)
        self._run_phase(fn=self._report, stage="report", job=self.config.job)

    def _load_data(self, job: StrategyJob):
        """Load data."""
        self.config.market_feed.fetch_data(symbols=job.tickers)

    def _generate_signals(self, job: StrategyJob):
        """Generate signals."""
        job.signals = self.config.strategy.generate_signals(
            job=job,
            market_snapshot=self.config.market_feed.market_snapshot,
            dates=self.config.market_feed.dates,
        )

    def _execute_trades(self, job: StrategyJob):
        """Run trade execution."""
        job.equity_curve = self.config.backtester.simulate_trade_execution(
            intents=job.signals,
            market_snapshot=self.config.market_feed.market_snapshot,
            dates=self.config.market_feed.dates,
        )

    def _run_backtest(self, job: StrategyJob):
        """Run backtest."""
        job.portfolio_snapshots = self.config.backtester.run(
            trades=job.equity_curve,
            market_snapshot=self.config.market_feed.market_snapshot,
            dates=self.config.market_feed.dates,
        )

    def _compute_metrics(self, job: StrategyJob):
        """Compute metrics."""
        job.metrics = self.config.metrics_engine.compute(
            benchmark_symbol=self.config.market_feed.data_inputs.benchmark_symbol,
            market_snapshot=self.config.market_feed.market_snapshot,
            snapshots=job.portfolio_snapshots,
            trades=job.equity_curve,
        )

    def _save_results(self, job: StrategyJob):
        """Save results."""
        if job.signals is not None:
            self.config.db.save("signals", job.signals, job)
        if job.equity_curve is not None:
            self.config.db.save("equity_curve", job.equity_curve, job)
        if job.metrics is not None:
            metrics_df = pd.DataFrame([job.metrics])
            self.config.db.save("metrics", metrics_df, job)

    def _report(self, job: StrategyJob):
        """Generate report."""
        self.config.reporter.report(
            job=job,
            market_snapshot=self.config.market_feed.market_snapshot,
            metrics=job.metrics,
        )
