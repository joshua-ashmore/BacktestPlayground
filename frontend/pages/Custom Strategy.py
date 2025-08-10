"""Run Custom Strategy."""

from datetime import date, timedelta

import streamlit as st

from components.execution.simple_engine import SimpleExecutionEngine
from components.job.base_model import StrategyJob
from components.market.market_data_feed import DataInputs, HistoricalFeed
from components.metrics.base_model import MetricsEngine
from engine.orchestrator import Orchestrator, OrchestratorConfig
from frontend.db_utils import metrics_to_dataframe, orchestrator_config_to_df_simple
from frontend.st_utils import display_trade_table, generate_charts, generate_layout
from frontend.strategy_util import STRATEGY_REGISTRY

st.title("Run Custom Strategy")


# --- 1. User Input ---
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        benchmark_options = {
            "^SPX": "S&P 500",
            "^NDX": "Nasdaq-100",
            "^DJI": "Dow Jones",
            "QQQ": "Nasdaq-100 ETF (QQQ)",
            "SPY": "S&P 500 ETF (SPY)",
            "VTI": "Total US Stock Market (VTI)",
            "^FTSE": "FTSE 100 (UK)",
            "^STOXX50E": "Euro Stoxx 50",
            "^N225": "Nikkei 225 (Japan)",
            "^HSI": "Hang Seng (Hong Kong)",
            "VEA": "FTSE Developed Markets ETF",
        }
        benchmark = st.selectbox(
            "Select Benchmark Index",
            options=benchmark_options.keys(),
            format_func=lambda k: benchmark_options[k],
        )
    with col2:
        start_date = st.date_input("Start Date", date(2025, 1, 1))
    with col3:
        end_date = st.date_input("End Date", date.today())


preset_tickers = {
    "AAPL": "Apple (Tech)",
    "MSFT": "Microsoft (Tech)",
    "AMZN": "Amazon (Consumer Discretionary)",
    "GOOG": "Google (Tech)",
    "META": "Meta (Communication Services)",
    "NVDA": "Nvidia (Tech)",
    "TSLA": "Tesla (Consumer Discretionary)",
    "BRK.B": "Berkshire Hathaway (Financials)",
    "JPM": "JPMorgan Chase (Financials)",
    "V": "Visa (Financials)",
    "UNH": "UnitedHealth (Health Care)",
    "XOM": "Exxon Mobil (Energy)",
    "CVX": "Chevron (Energy)",
    "PFE": "Pfizer (Health Care)",
    "WMT": "Walmart (Consumer Staples)",
    "COST": "Costco (Consumer Staples)",
    "HD": "Home Depot (Consumer Discretionary)",
    "INTC": "Intel (Tech)",
    "CSCO": "Cisco (Tech)",
    "NKE": "Nike (Consumer Discretionary)",
}

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_presets = st.multiselect(
            "Select from Preset Tickers",
            options=list(preset_tickers.keys()),
            default=list(preset_tickers.keys())[:1],
            format_func=lambda x: preset_tickers.get(x, x),
            help="Choose from a diversified set of high-quality U.S. stocks.",
        )
    with col2:
        custom_tickers = st.text_input(
            "Add Custom Tickers (comma-separated)",
            help="Manually enter tickers not in the preset list (e.g., BABA, CRM, IBM)",
        )

    # Merge and clean input
    custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    tickers = sorted(set(selected_presets + custom_list))

    if tickers:
        st.write(f"Final Ticker List: {', '.join(tickers)}")


with st.container():
    st.subheader("Strategy Parameter Selection")
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=list(STRATEGY_REGISTRY.keys()),
    )
    st.info(STRATEGY_REGISTRY[selected_strategy]["help"])

    if not tickers:
        st.warning("Please select at least one ticker before running the strategy.")
        st.stop()
    strategy_config = STRATEGY_REGISTRY[selected_strategy]
    strategy_kwargs = strategy_config["ui_func"]()

with st.container():
    st.subheader("Backtester Parameters")

    col1, col2 = st.columns(2)

    with col1:
        initial_cash = st.number_input(
            "Initial Cash (£)",
            value=100000,
            min_value=0,
            step=1000,
            help="Starting capital for the backtest",
        )
        max_hold_days = st.slider(
            "Max Hold Days",
            value=60,
            min_value=1,
            max_value=365,
            help="Maximum number of days to hold a position",
        )

    with col2:
        allocation_pct_per_trade = st.slider(
            "Allocation % per Trade",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="Percentage of cash allocated per trade",
        )
        allocation_pct_per_trade /= 100
        take_profit_pct = st.slider(
            "Take Profit %",
            min_value=0,
            max_value=100,
            value=6,
            step=1,
            help="Profit percentage to take profit at",
        )
        take_profit_pct /= 100
        stop_loss_pct = st.slider(
            "Stop Loss %",
            min_value=-10,
            max_value=0,
            value=-3,
            step=1,
            help="Loss percentage to trigger stop loss",
        )
        stop_loss_pct /= 100

run_btn = st.button("Run Strategy")

# --- 2. Run the Strategy ---
if run_btn:
    with st.spinner("Running backtest..."):
        strategy_class = strategy_config["class"]
        strategy_instance = strategy_class(symbols=tickers, **strategy_kwargs)
        config = OrchestratorConfig(
            start_date=start_date,
            job=StrategyJob(
                job_name="Custom", current_date=start_date, tickers=tickers
            ),
            market_feed=HistoricalFeed(
                data_inputs=DataInputs(
                    benchmark_symbol=benchmark,
                    end_date=end_date + timedelta(days=1),
                ),
                source="yahoo",
            ),
            strategy=strategy_instance,
            execution_engine=SimpleExecutionEngine(
                initial_cash=initial_cash,
                max_hold_days=max_hold_days,
                allocation_pct_per_trade=allocation_pct_per_trade,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
            ),
            metrics_engine=MetricsEngine(save_metrics=False),
        )
        orchestrator = Orchestrator(config=config)
        orchestrator.run()

    # --- 3. Show Results ---
    selected_summary = orchestrator_config_to_df_simple(
        metrics=config.job.metrics
    ).iloc[0]

    if not config.job.signals:
        st.warning("No signals returned for configured strategy.")
        st.stop()
    else:
        st.success("Backtest complete!")
        generate_layout(selected_summary=selected_summary)

        ts_df = metrics_to_dataframe(metrics=orchestrator.config.job.metrics)
        generate_charts(ts_df=ts_df, regime=False)

        display_trade_table(trades=orchestrator.config.job.equity_curve)
