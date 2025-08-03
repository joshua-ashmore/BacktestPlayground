"""Orchestrator."""

from datetime import date
from typing import Callable, List, Optional

import pandas as pd
from pydantic import BaseModel, model_validator

from backtester.market_data.market_data_feed import MarketFeed
from backtester.market_data.sqlite_market_data_cache import SQLiteMetricsStore
from components.backtester.all_backtesters import Backtesters
from components.job.base_model import StrategyJob
from components.metrics.base_model import MetricsEngine, PortfolioMetrics
from components.regime.regime_engines import RegimeEngines
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
    backtester: Backtesters
    metrics_engine: MetricsEngine
    strategy: Optional[Strategies] = None
    regime_engine: Optional[RegimeEngines] = None
    reporter: Optional[PDFReporter] = None
    all_tickers: List[str] = None
    hooks: List[Hook] = []

    start_date: Optional[date] = None

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        if not self.strategy and not self.regime_engine:
            raise ValueError("One of strategy or regime engine must be provided.")
        if not self.all_tickers:
            if self.strategy:
                self.all_tickers = self.strategy.symbols
            else:
                self.all_tickers = [
                    symbol
                    for strategy in self.regime_engine.strategy_map.values()
                    for symbol in strategy.symbols
                ]
        if not self.start_date:
            self.start_date = min(self.market_feed.dates)
        return self


class Orchestrator(BaseModel):
    """Orchestrator."""

    config: OrchestratorConfig

    def _run_phase(
        self,
        fn: Callable[[StrategyJob], None],
        stage: str,
        job: StrategyJob,
        target_date: Optional[date] = None,
        previous_date: Optional[date] = None,
    ):
        for hook in self.config.hooks:
            hook.before(stage=stage, job=job)
        if target_date and previous_date:
            fn(job=job, target_date=target_date, previous_date=previous_date)
        elif target_date:
            fn(job=job, target_date=target_date)
        else:
            fn(job=job)
        for hook in self.config.hooks:
            hook.after(stage=stage, job=job)

    def run(self) -> PortfolioMetrics | None:
        """Run strategy."""
        self._run_phase(fn=self._pre_save, stage="pre_save", job=self.config.job)
        self._run_phase(fn=self._load_data, stage="load_data", job=self.config.job)
        dates = [
            _date
            for _date in self.config.market_feed.dates
            if _date >= self.config.start_date
        ]
        for i, target_date in enumerate(dates):
            if i == 0:
                continue
            previous_date = dates[i - 1]
            self._run_phase(
                fn=self._detect_regime,
                stage="detect_regime",
                job=self.config.job,
                target_date=target_date,
                previous_date=previous_date,
            )
            self._run_phase(
                fn=self._generate_signals,
                stage="generate_signals",
                job=self.config.job,
                target_date=target_date,
                previous_date=previous_date,
            )
            self._run_phase(
                fn=self._execute_trades,
                stage="execute_trades",
                job=self.config.job,
                target_date=target_date,
                previous_date=previous_date,
            )
            self._run_phase(
                fn=self._run_backtest,
                stage="backtest",
                job=self.config.job,
                target_date=target_date,
            )
        self._run_phase(fn=self._compute_metrics, stage="metrics", job=self.config.job)
        self._run_phase(fn=self._save_results, stage="save", job=self.config.job)
        self._run_phase(fn=self._report, stage="report", job=self.config.job)

    def _pre_save(self, job: StrategyJob):
        """Pre-save data."""
        metrics_store = SQLiteMetricsStore()
        if self.config:
            metrics_store.save_strategy_config(
                strategy_name=job.job_name, config=self.config
            )

    def _load_data(self, job: StrategyJob):
        """Load data."""
        job.tickers = self.config.all_tickers
        self.config.market_feed.fetch_data(symbols=job.tickers)
        job.market_snapshot = self.config.market_feed.market_snapshot

    def _detect_regime(
        self,
        job: StrategyJob,
        target_date: date,
        previous_date: date,
    ):
        """Detect current market regime and assign strategy accordingly."""
        if self.config.regime_engine:
            self.config.strategy = self.config.regime_engine.detect_regime(
                job=job,
                target_date=target_date,
                previous_date=previous_date,
            )

    def _generate_signals(
        self, job: StrategyJob, target_date: date, previous_date: date
    ):
        """Generate signals."""
        job.signals.extend(
            self.config.strategy.generate_signal_on_date(
                job=job, target_date=target_date, previous_date=previous_date
            )
        )

    def _execute_trades(self, job: StrategyJob, target_date: date, previous_date: date):
        """Run trade execution."""
        job.equity_curve.extend(
            self.config.backtester.simulate_trade_execution_on_date(
                job=job,
                target_date=target_date,
                previous_date=previous_date,
                strategy=self.config.strategy,
            )
        )

    def _run_backtest(self, job: StrategyJob, target_date: date):
        """Run backtest."""
        job.portfolio_snapshots.append(
            self.config.backtester.run_on_date(job=job, target_date=target_date)
        )

    def _compute_metrics(self, job: StrategyJob):
        """Compute metrics."""
        job.metrics = self.config.metrics_engine.compute(
            benchmark_symbol=self.config.market_feed.data_inputs.benchmark_symbol,
            market_snapshot=self.config.market_feed.market_snapshot,
            snapshots=job.portfolio_snapshots,
            trades=job.equity_curve,
            regime_engine=self.config.regime_engine,
        )

    def _save_results(self, job: StrategyJob) -> PortfolioMetrics | None:
        """Save results."""
        metrics_store = SQLiteMetricsStore()
        if job.metrics is not None and self.config.metrics_engine.save_metrics:
            metrics_store.save_portfolio_metrics(
                metrics=job.metrics, strategy_name=job.job_name
            )

    def _report(self, job: StrategyJob):
        """Generate report."""
        if self.config.reporter:
            self.config.reporter.report(
                job=job,
                market_snapshot=self.config.market_feed.market_snapshot,
                metrics=job.metrics,
                min_date=self.config.start_date,
            )
