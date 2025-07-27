"""Example Config."""

import datetime

MOMENTUM_CONFIG = {
    "job": {
        "strategy_name": "Test Momentum",
        "current_date": datetime.date(2025, 7, 22),
        "tickers": [
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
        "signals": None,
        "equity_curve": None,
        "portfolio_snapshots": None,
        "metrics": None,
    },
    "market_feed": {
        "data_inputs": {
            "benchmark_symbol": "^SPX",
            "start_date": datetime.date(2020, 1, 1),
            "end_date": datetime.date(2025, 7, 23),
            "symbols": None,
        },
        "source": "yahoo",
        "market_snapshot": None,
        "dates": None,
        "feed_type": "historical",
    },
    "strategy": {"strategy_name": "momentum", "base_window": 60, "top_n": 3},
    "backtester": {
        "backtester_type": "simple",
        "initial_cash": 100000,
        "cash": 100000,
        "portfolio_value_history": {},
        "position_book": {},
        "max_hold_days": 60,
        "allocation_pct_per_trade": 0.3,
        "stop_loss_pct": -0.03,
        "take_profit_pct": 100.0,
    },
    "metrics_engine": {"rolling_window": 20},
    "reporter": {"output_dir": "reports", "logo_path": "logo.png", "styles": None},
    "hooks": [],
}
