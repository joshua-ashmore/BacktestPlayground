"""Streamlit Utils."""

from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from components.trades.trade_model import Trade
from engine.orchestrator import OrchestratorConfig


def generate_config(config: OrchestratorConfig):
    """Generate Strategy Config."""

    st.subheader("Strategy Configuration")

    with st.expander("Job Details"):
        st.write(f"**Strategy Name**: {config.job.strategy_name}")
        st.write(f"**Date**: {config.job.current_date}")
        st.write(f"**Tickers**: {', '.join(config.job.tickers)}")

    with st.expander("Backtester Settings"):
        st.write(f"**Initial Cash**: ${config.backtester.initial_cash:,}")
        st.write(f"**Max Hold Days**: {config.backtester.max_hold_days}")
        st.write(
            f"**Allocation per Trade**: {config.backtester.allocation_pct_per_trade * 100:.1f}%"
        )
        st.write(f"**Stop Loss**: {config.backtester.stop_loss_pct * 100:.1f}%")
        st.write(f"**Take Profit**: {config.backtester.take_profit_pct * 100:.1f}%")

    with st.expander("Market Feed"):
        st.write(f"**Source**: {config.market_feed.source.value}")
        st.write(f"**Start Date**: {config.market_feed.data_inputs.start_date}")
        st.write(f"**End Date**: {config.market_feed.data_inputs.end_date}")
        st.write(f"**Benchmark**: {config.market_feed.data_inputs.benchmark_symbol}")


def generate_layout(selected_summary):
    """Generate Layout."""
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Cumulative Return", f"{selected_summary['cumulative_return']:.2%}")
    col2.metric(
        "Annualized Volatility", f"{selected_summary['annualized_volatility']:.2%}"
    )
    col3.metric("Sharpe Ratio", f"{selected_summary['sharpe_ratio']:.2f}")
    col4.metric("Information Ratio", f"{selected_summary['information_ratio']:.2f}")
    col5.metric("Max Drawdown", f"{selected_summary['max_drawdown']:.2%}")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Annualized Return", f"{selected_summary['annualized_return']:.2%}")
    col7.metric("Win Rate", f"{selected_summary['win_rate']:.2%}")
    col8.metric("Num Trades", f"{int(selected_summary['num_trades'] or 0)}")
    col9.metric("Avg PnL", f"{selected_summary['average_pnl']:.2f}")
    col10.metric("Turnover", f"{selected_summary['turnover']:.2f}")


def generate_charts(ts_df: pd.DataFrame):
    """Generate Charts."""

    if ts_df.empty:
        st.warning("No time series data found for this run.")
    else:
        st.subheader("Equity Curve")

        min_equity = ts_df["equity_curve"].min()
        line_chart = (
            alt.Chart(ts_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelAngle=-45,
                    ),
                ),
                y=alt.Y(
                    "equity_curve:Q",
                    title="Equity Curve Value",
                    scale=alt.Scale(domainMin=min_equity),
                ),
                tooltip=["date:T", "equity_curve:Q"],
            )
            .properties(height=300, width="container")
        )

        st.altair_chart(line_chart, use_container_width=True)

        st.subheader("Rolling Sharpe Ratio")
        sharpe_chart = (
            alt.Chart(ts_df)
            .mark_line(color="orange")
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelAngle=-45,
                    ),
                ),
                y=alt.Y("rolling_sharpe:Q", title="Sharpe"),
                tooltip=["date:T", "rolling_sharpe:Q"],
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(sharpe_chart, use_container_width=True)

        st.subheader("Rolling Drawdown")
        drawdown_chart = (
            alt.Chart(ts_df)
            .mark_area(color="red", opacity=0.4)
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelAngle=-45,
                    ),
                ),
                y=alt.Y("rolling_drawdown:Q", title="Drawdown"),
                tooltip=["date:T", "rolling_drawdown:Q"],
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(drawdown_chart, use_container_width=True)

        st.subheader("Histogram of Daily Returns")
        hist = (
            alt.Chart(ts_df)
            .mark_bar()
            .encode(
                alt.X("daily_return:Q", bin=alt.Bin(maxbins=50), title="Daily Return"),
                y="count()",
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(hist, use_container_width=True)


def display_trade_table(trades: List[Trade]):
    if not trades:
        st.info("No trades found for this strategy.")
        return

    # Convert list of Trade objects to DataFrame
    df = pd.DataFrame([t.model_dump() for t in trades])

    # Optional formatting (you can remove/adjust as needed)
    df["open_date"] = pd.to_datetime(df["open_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    df["notional"] = df["notional"].round(2)
    df["pnl"] = df["pnl"].round(2)
    df["price"] = df["price"].round(2)
    df["close_price"] = df["close_price"].round(2)
    df["quantity"] = df["quantity"].round(2)

    df = df.rename(
        columns={
            "symbol": "Ticker",
            "open_date": "Open Date",
            "close_date": "Close Date",
            "quantity": "Qty",
            "price": "Open Price",
            "close_price": "Close Price",
            "notional": "Notional",
            "side": "Side",
            "pnl": "PnL",
        }
    )

    st.subheader("Trade History")
    st.dataframe(df, use_container_width=True)
