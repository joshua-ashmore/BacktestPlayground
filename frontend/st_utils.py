"""Streamlit Utils."""

from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from components.trades.trade_model import Trade
from engine.orchestrator import OrchestratorConfig
from frontend.db_utils import load_strategy_metrics


def generate_config(config: OrchestratorConfig):
    """Generate Strategy Config."""

    st.subheader("Strategy Configuration")

    # Create 3 columns
    col1, col2, col3 = st.columns(3)

    # --- Column 1: Job Details ---
    with col1.expander("Job Details"):
        st.write(f"**Strategy Name**: {config.job.job_name}")
        st.write(f"**Date**: {config.job.current_date}")
        if config.regime_engine:
            tickers = ", ".join(
                symbol
                for job in config.regime_engine.strategy_map.values()
                for symbol in job.symbols
            )
        else:
            tickers = ", ".join(config.job.tickers)
        st.write(f"**Tickers**: {tickers}")

    # --- Column 2: Backtester Settings ---
    with col2.expander("Backtester Settings"):
        st.write(f"**Initial Cash**: ${config.execution_engine.initial_cash:,}")
        st.write(f"**Max Hold Days**: {config.execution_engine.max_hold_days}")
        st.write(
            f"**Allocation per Trade**: {config.execution_engine.allocation_pct_per_trade * 100:.1f}%"
        )
        st.write(f"**Stop Loss**: {config.execution_engine.stop_loss_pct * 100:.1f}%")
        st.write(
            f"**Take Profit**: {config.execution_engine.take_profit_pct * 100:.1f}%"
        )

    # --- Column 3: Market Feed ---
    with col3.expander("Market Feed"):
        st.write(f"**Source**: {config.market_feed.source.value}")
        st.write(f"**Start Date**: {config.market_feed.data_inputs.start_date}")
        st.write(f"**End Date**: {config.market_feed.data_inputs.end_date}")
        st.write(f"**Benchmark**: {config.market_feed.data_inputs.benchmark_symbol}")


def generate_layout(selected_summary):
    """Generate Layout."""
    st.subheader("Portfolio Metrics")
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


def generate_charts(ts_df: pd.DataFrame, regime: bool = True):
    """Generate Charts."""

    # if ts_df.empty:
    #     st.warning("No time series data found for this run.")
    # else:
    #     st.subheader("Equity Curve")

    #     print(ts_df)
    #     ts_df["regime"] = ts_df["regime_timeseries"]
    #     ts_df["prev_regime"] = ts_df["regime"].shift(1)
    #     ts_df["segment"] = (ts_df["regime"] != ts_df["prev_regime"]).cumsum()

    #     regime_ranges = (
    #         ts_df.groupby(["segment", "regime"])
    #         .agg(start_date=("date", "min"), end_date=("date", "max"))
    #         .reset_index()
    #     )
    #     color_scale = alt.Scale(
    #         domain=["trending", "volatile", "mean_reverting"],
    #         range=["#1f77b4", "#ff7f0e", "#2ca02c"],  # vibrant blue, orange, green
    #     )

    #     regime_chart = (
    #         alt.Chart(regime_ranges)
    #         .mark_rect(opacity=0.7)
    #         .encode(
    #             x="start_date:T",
    #             x2="end_date:T",
    #             color=alt.Color(
    #                 "regime:N", scale=color_scale, legend=alt.Legend(title="Regime")
    #             ),
    #         )
    #     )

    #     min_equity = ts_df["equity_curve"].min()
    #     line_chart = (
    #         alt.Chart(ts_df)
    #         .mark_line()
    #         .encode(
    #             x=alt.X(
    #                 "date:T",
    #                 title="Date",
    #                 axis=alt.Axis(format="%b %Y", labelAngle=-45),
    #             ),
    #             y=alt.Y(
    #                 "equity_curve:Q",
    #                 title="Equity Curve Value",
    #                 scale=alt.Scale(domainMin=min_equity),
    #             ),
    #             tooltip=["date:T", "equity_curve:Q"],
    #         )
    #         .properties(height=300, width="container")
    #     )
    #     if regime:
    #         combined_chart = (
    #             alt.layer(regime_chart, line_chart)
    #             .resolve_scale(color="independent")
    #             .interactive()
    #         )
    #     else:
    #         combined_chart = line_chart

    #     st.altair_chart(combined_chart, use_container_width=True)

    if ts_df.empty:
        st.warning("No time series data found for this run.")
        return

    st.subheader("Equity Curve")

    if regime:
        print(ts_df["regime_timeseries"])
        ts_df["regime"] = ts_df["regime_timeseries"]
        ts_df["segment"] = (ts_df["regime"] != ts_df["regime"].shift(1)).cumsum()

        # Build regime rectangles
        regime_ranges = (
            ts_df.groupby(["segment", "regime"])
            .agg(start_date=("date", "min"), end_date=("date", "max"))
            .reset_index()
        )

        color_scale = alt.Scale(
            domain=["trending", "volatile", "mean_reverting"],
            range=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )

        regime_chart = (
            alt.Chart(regime_ranges)
            .mark_rect(opacity=0.15)  # lighter so line is visible
            .encode(
                x="start_date:T",
                x2="end_date:T",
                color=alt.Color(
                    "regime:N", scale=color_scale, legend=alt.Legend(title="Regime")
                ),
            )
        )
        # ts_df["regime"] = ts_df["regime_timeseries"]
        # ts_df["prev_regime"] = ts_df["regime"].shift(1)
        # ts_df["segment"] = (ts_df["regime"] != ts_df["prev_regime"]).cumsum()

        # regime_ranges = (
        #     ts_df.groupby(["segment", "regime"])
        #     .agg(start_date=("date", "min"), end_date=("date", "max"))
        #     .reset_index()
        # )

        # color_scale = alt.Scale(
        #     domain=["trending", "volatile", "mean_reverting"],
        #     range=["#1f77b4", "#ff7f0e", "#2ca02c"],  # blue, orange, green
        # )

        # regime_chart = (
        #     alt.Chart(regime_ranges)
        #     .mark_rect(opacity=0.3)
        #     .encode(
        #         x=alt.X("start_date:T", title="Date"),
        #         x2="end_date:T",
        #         color=alt.Color(
        #             "regime:N", scale=color_scale, legend=alt.Legend(title="Regime")
        #         ),
        #     )
        # )
    else:
        regime_chart = None

    min_equity = ts_df["equity_curve"].min()

    line_chart = (
        alt.Chart(ts_df)
        .mark_line()
        .encode(
            x=alt.X(
                "date:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-45)
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

    if regime_chart is not None:
        combined_chart = (
            alt.layer(regime_chart, line_chart)
            .resolve_scale(color="independent")
            .interactive()
        )
    else:
        combined_chart = line_chart.interactive()

    st.altair_chart(combined_chart, use_container_width=True)

    # TODO: implement this but with scaling items from above
    # # --- Step 1: Create smoothed segments ---
    # ts_df = ts_df.copy()
    # ts_df["prev_regime"] = ts_df["regime_timeseries"].shift()
    # ts_df["segment"] = (ts_df["regime_timeseries"] != ts_df["prev_regime"]).cumsum()

    # # Group start/end for each segment
    # regime_ranges = (
    #     ts_df.groupby("segment")
    #     .agg(
    #         start_date=("date", "first"),
    #         end_date=("date", "last"),
    #         regime=("regime_timeseries", "first"),
    #     )
    #     .reset_index(drop=True)
    # )

    # # --- Step 2: Merge short-lived segments ---
    # min_days = 3
    # merged_ranges = []
    # prev_range = None

    # for _, row in regime_ranges.iterrows():
    #     duration = (row["end_date"] - row["start_date"]).days + 1
    #     if duration < min_days and prev_range is not None:
    #         # Merge with previous
    #         prev_range["end_date"] = row["end_date"]
    #     else:
    #         if prev_range:
    #             merged_ranges.append(prev_range)
    #         prev_range = row.to_dict()

    # if prev_range:
    #     merged_ranges.append(prev_range)

    # regime_ranges = pd.DataFrame(merged_ranges)

    # # --- Step 3: Build equity chart ---
    # min_equity = ts_df["equity_curve"].min()

    # equity_chart = (
    #     alt.Chart(ts_df)
    #     .mark_line()
    #     .encode(
    #         x=alt.X(
    #             "date:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-45)
    #         ),
    #         y=alt.Y(
    #             "equity_curve:Q",
    #             title="Equity Curve Value",
    #             scale=alt.Scale(domainMin=min_equity),
    #         ),
    #         tooltip=["date:T", "equity_curve:Q"],
    #     )
    #     .properties(height=300, width="container")
    # )

    # # --- Step 4: Build regime background ---
    # if regime:
    #     regime_chart = (
    #         alt.Chart(regime_ranges)
    #         .mark_rect(opacity=0.45)
    #         .encode(
    #             x="start_date:T",
    #             x2="end_date:T",
    #             color=alt.Color(
    #                 "regime:N",
    #                 scale=alt.Scale(scheme="category10"),
    #                 legend=alt.Legend(title="Regime"),
    #             ),
    #         )
    #     )
    #     chart = alt.layer(regime_chart, equity_chart)
    # else:
    #     chart = equity_chart

    # st.altair_chart(chart, use_container_width=True)

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

    # Flatten all legs from all trades into a list of dicts
    leg_records = []
    for trade in trades:
        for leg in trade.legs:
            leg_records.append(
                {
                    "symbol": leg.symbol,
                    "open_date": pd.to_datetime(leg.open_date),
                    "close_date": pd.to_datetime(leg.close_date),
                    "quantity": round(leg.quantity, 2),
                    "price": round(leg.price, 2),
                    "close_price": (
                        round(leg.close_price, 2) if leg.close_price else None
                    ),
                    "notional": round(leg.notional, 2),
                    "side": leg.side,
                    "pnl": round(leg.pnl, 2) if leg.pnl else None,
                }
            )

    if not leg_records:
        st.info("No trade legs found.")
        return

    df = pd.DataFrame(leg_records)

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


def generate_multi_strat_table(selected_summary_id):
    st.subheader("Multi Strategy Metrics")
    strategy_metrics = load_strategy_metrics(summary_id=selected_summary_id)
    # Convert dict to DataFrame
    df = pd.DataFrame.from_dict(strategy_metrics, orient="index")

    # Rename the indices nicely
    df.index = df.index.str.replace("_", " ").str.title()
    df.index.name = "Strategy"

    # Format columns
    df["Win Rate"] = df["win_rate"].map("{:.1%}".format)
    df["Avg PnL"] = df["avg_pnl"].map("{:.2f}".format)
    df["Total PnL"] = df["total_pnl"].map("{:.2f}".format)
    df["Num Trades"] = df["num_trades"].astype(int)

    # Drop old columns now replaced with formatted ones
    df = df.drop(columns=["win_rate", "avg_pnl", "total_pnl", "num_trades"])

    # Reorder columns nicely
    df = df[["Num Trades", "Win Rate", "Avg PnL", "Total PnL"]]

    def style_table(styler):
        # Color text for Num Trades and Avg PnL columns based on value
        def color_text(val):
            try:
                val_float = float(val.replace(",", "").replace("%", ""))
                if val_float > 0:
                    return "color: green; font-weight: bold"
                elif val_float < 0:
                    return "color: red; font-weight: bold"
                else:
                    return ""
            except BaseException:
                return ""

        # Apply white color and center text for all cells
        styler = styler.set_properties(**{"color": "white", "text-align": "center"})

        # Apply text coloring only to specific columns
        styler = styler.applymap(color_text, subset=["Num Trades", "Avg PnL"])

        return styler

    st.dataframe(style_table(df.style))
