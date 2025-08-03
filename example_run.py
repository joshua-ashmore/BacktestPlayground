"""Example Script."""

from datetime import date, timedelta

from backtester.market_data.market_data_feed import DataInputs, HistoricalFeed
from components.backtester.simple_backtester import SimpleBacktester
from components.job.base_model import StrategyJob
from components.metrics.base_model import MetricsEngine
from components.regime.multi_hmm_engine import MultiHMMEngine
from components.reporter.base_model import PDFReporter
from components.strategies.angular_momentum import AngleMomentumStrategy
from components.strategies.mean_reversion import PairsTradingStrategyJob
from components.strategies.volatility_breakout import VolatilityBreakoutStrategyJob
from engine.orchestrator import Orchestrator, OrchestratorConfig

mean_reverting_strategy = PairsTradingStrategyJob(
    symbols=[
        # Consumer Staples
        "KO",
        "PEP",
        "WMT",
        "COST",
        "TGT",
        "KR",
        "CL",
        "PG",
        "KMB",
        "GIS",
        "MO",
        "PM",
        # Healthcare
        "JNJ",
        "PFE",
        "MRK",
        "ABT",
        "BMY",
        "GILD",
        "LLY",
        "AMGN",
        # Banks
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "HAL",
        # Utilities
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
    ]
)
# mean_reverting_strategy = FallbackStrategyJob()
volatility_strategy = VolatilityBreakoutStrategyJob(
    symbols=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]
)
trending_strategy = AngleMomentumStrategy(
    max_holding_days=10,
    symbols=[
        "NVDA",
        "TSLA",
        "AAPL",
        "MSFT",
        "AMD",
        "META",
        "AMZN",
    ],
)

last_date = date(2025, 7, 22)
orchestrator_config = OrchestratorConfig(
    market_feed=HistoricalFeed(
        data_inputs=DataInputs(
            benchmark_symbol="^SPX",
            start_date=date(2019, 1, 1),
            end_date=last_date + timedelta(days=1),
        ),
        feed_type="historical",
        source="yahoo",
    ),
    backtester=SimpleBacktester(
        stop_loss_pct=-0.05,
        take_profit_pct=0.2,
        max_hold_days=60,
        allocation_pct_per_trade=0.3,
    ),
    metrics_engine=MetricsEngine(rolling_window=20, save_metrics=True),
    regime_engine=MultiHMMEngine(
        strategy_map={
            "trending": trending_strategy,
            "volatile": volatility_strategy,
            "mean_reverting": mean_reverting_strategy,
        },
    ),
    reporter=PDFReporter(output_dir="reports"),
    job=StrategyJob(
        job_name="Regime Detection 3Y Multi HMM (All Regimes)",
        current_date=last_date,
    ),
    start_date=date(2025, 1, 1),
)
orchestrator = Orchestrator(config=orchestrator_config)
orchestrator.run()
b = 1
