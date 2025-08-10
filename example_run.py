"""Example Script."""

from datetime import date

from components.execution.simple_engine import SimpleExecutionEngine
from components.job.base_model import StrategyJob
from components.market.market_data_feed import DataInputs, HistoricalFeed
from components.metrics.base_model import MetricsEngine
from components.regime.multi_hmm_engine import MultiHMMEngine  # noqa: F401
from components.reporter.base_model import PDFReporter
from components.strategies.fallback_strategy import FallbackStrategyJob  # noqa: F401
from components.strategies.gradient_momentum import GradientMomentumStrategy
from components.strategies.mean_reversion import PairsTradingStrategyJob  # noqa: F401
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
mean_reverting_strategy = FallbackStrategyJob()
volatility_strategy = VolatilityBreakoutStrategyJob(
    symbols=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]  # "^SPX"
)
# volatility_strategy = FallbackStrategyJob()
trending_strategy = GradientMomentumStrategy(
    max_holding_days=10,
    rsi_lower=30,
    rsi_upper=70,
    angle_calculation_day_offset=30,
    # up_trend_angle_threshold_deg=30,
    # down_trend_angle_threshold_deg=-30,
    # angle_calculation_day_offset=10,
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

last_date = date.today()
orchestrator_config = OrchestratorConfig(
    market_feed=HistoricalFeed(
        data_inputs=DataInputs(
            benchmark_symbol="^SPX",
            # benchmark_symbol="BTC-USD",
        ),
        feed_type="historical",
        source="yahoo",
    ),
    execution_engine=SimpleExecutionEngine(
        stop_loss_pct=-0.05,
        take_profit_pct=0.2,
        max_hold_days=60,
        allocation_pct_per_trade=0.3,
    ),
    metrics_engine=MetricsEngine(rolling_window=20, save_metrics=True),
    # strategy=trending_strategy,
    # strategy=volatility_strategy,
    # strategy=mean_reverting_strategy,
    regime_engine=MultiHMMEngine(
        strategy_map={
            "trending": trending_strategy,
            "volatile": volatility_strategy,
            "mean_reverting": mean_reverting_strategy,
        },
        preferred_state_idx={
            "trending": 0,
            "volatile": 1,
            "mean_reverting": 2,
        },
    ),
    reporter=PDFReporter(output_dir="reports"),
    job=StrategyJob(
        # job_name="Gradient Momentum",
        # job_name="Volatility Breakout",
        # job_name="Mean Reversion",
        job_name="Regime Detection HMM (2 Regimes)",
        current_date=last_date,
    ),
    start_date=date(2025, 1, 1),
)
orchestrator = Orchestrator(config=orchestrator_config)
orchestrator.run()
b = 1
