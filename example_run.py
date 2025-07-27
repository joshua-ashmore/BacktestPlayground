"""Example Script."""

from datetime import date, timedelta

from backtester.market_data.market_data_feed import DataInputs, HistoricalFeed
from components.backtester.simple_backtester import SimpleBacktester
from components.job.base_model import StrategyJob
from components.metrics.base_model import MetricsEngine
from components.reporter.base_model import PDFReporter
from components.strategies.momentum_strategy import MomentumStrategyJob
from engine.orchestrator import Orchestrator, OrchestratorConfig


last_date = date(2025, 7, 22)
orchestrator_config = OrchestratorConfig(
    market_feed=HistoricalFeed(
        data_inputs=DataInputs(
            benchmark_symbol="^SPX",
            start_date=date(2020, 1, 1),
            end_date=last_date + timedelta(days=1),
        ),
        feed_type="historical",
        source="yahoo",
    ),
    strategy=MomentumStrategyJob(base_window=60),
    backtester=SimpleBacktester(
        stop_loss_pct=-0.03,
        take_profit_pct=100,
        max_hold_days=60,
        allocation_pct_per_trade=0.3,
    ),
    metrics_engine=MetricsEngine(rolling_window=20),
    reporter=PDFReporter(output_dir="reports"),
    job=StrategyJob(
        strategy_name="Test Momentum",
        current_date=last_date,
        tickers=[
            "TSLA",
            "MSFT",
            "AAPL",
            "AMZN",
            "NVDA",
            "AMD",
            "GOOG",
            "META",
            "JPM",
            "BRK.B",
            "UNH",
            "XOM",
            "PFE",
            "V",
            "WMT",
            "COST",
        ],
    ),
)
orchestrator_results = Orchestrator(config=orchestrator_config).run()
