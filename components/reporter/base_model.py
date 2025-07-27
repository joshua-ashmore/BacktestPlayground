"""Reporter Base Model."""

from datetime import datetime
import io
import os
from typing import Any, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from pydantic import BaseModel

from backtester.market_data.market import MarketSnapshot
from components.job.base_model import StrategyJob

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from components.metrics.base_model import PortfolioMetrics
from components.trades.trade_model import Trade


class PDFReporter(BaseModel):
    """PDF Reporter Model."""

    output_dir: str = "reports"
    logo_path: str = "logo.png"
    styles: Optional[Any] = None

    def report(
        self,
        job: "StrategyJob",
        market_snapshot: MarketSnapshot,
        metrics: PortfolioMetrics,
    ):
        """Report."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

        filename = f"{job.strategy_name}_{metrics.start_date}_{metrics.end_date}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []

        # Header with logo and title
        # logo = Image(self.logo_path, width=0.8 * inch, height=0.8 * inch)
        title = Paragraph(
            f"<b>Strategy Report: {job.strategy_name}</b>", self.styles["Title"]
        )
        date_info = Paragraph(
            f"Run Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles["Normal"]
        )
        # elements.extend([logo, title, date_info, Spacer(1, 0.2 * inch)])
        elements.extend([title, date_info, Spacer(1, 0.2 * inch)])

        # Strategy Description
        # elements.append(
        #     Paragraph(
        #         "This report summarizes the performance of the strategy over the selected backtest period. "
        #         "The metrics below offer insight into the risk-adjusted performance, capital deployment, "
        #         "and trading efficiency.",
        #         self.styles["Normal"],
        #     )
        # )
        # elements.append(Spacer(1, 0.2 * inch))

        # Metadata table
        metadata_data = [
            ["Start Date", str(metrics.start_date)],
            ["End Date", str(metrics.end_date)],
            ["Total Return", f"{metrics.cumulative_return:.2%}"],
            ["Annualized Return", f"{metrics.annualized_return:.2%}"],
            ["Annualized Volatility", f"{metrics.annualized_volatility:.2%}"],
            ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
            ["Information Ratio", f"{metrics.information_ratio:.2f}"],
            ["Max Drawdown", f"{metrics.max_drawdown:.2%}"],
            ["Trades Executed", str(metrics.num_trades or "N/A")],
            [
                "Win Rate",
                f"{(metrics.win_rate * 100):.1f}%" if metrics.win_rate else "N/A",
            ],
            [
                "Avg Trade PnL",
                f"{metrics.average_pnl:.2f}" if metrics.average_pnl else "N/A",
            ],
        ]

        metadata_table = Table(metadata_data)
        metadata_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]
            )
        )
        # elements.append(metadata_table)
        # elements.append(Spacer(1, 0.3 * inch))

        summary_paragraph = Paragraph(
            f"""
            <b>Strategy Summary:</b><br/>
            {job.strategy_name} executed over {metrics.start_date} to {metrics.end_date}.<br/>
            Achieved {metrics.cumulative_return:.2%} return with a Sharpe ratio of {metrics.sharpe_ratio:.2f}.<br/>
            Max drawdown of {metrics.max_drawdown:.2%}.<br/>
            {metrics.num_trades} trades executed with a win rate of {0}%. metrics.win_rate * 100:.1f
            """,
            self.styles["Normal"],
        )

        side_by_side_table = Table(
            [[metadata_table, summary_paragraph]], colWidths=[3.5 * inch, 3.5 * inch]
        )
        side_by_side_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

        elements.append(side_by_side_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Commentary Section
        # commentary = []
        # if metrics.sharpe_ratio > 2:
        #     commentary.append("Exceptional risk-adjusted returns.")
        # elif metrics.sharpe_ratio > 1:
        #     commentary.append("Strong Sharpe ratio.")
        # else:
        #     commentary.append("Sharpe ratio indicates room for improvement.")

        # if metrics.max_drawdown > 0.10:
        #     commentary.append("Significant drawdowns observed.")
        # if metrics.cumulative_return > 0.20:
        #     commentary.append("Healthy overall performance.")

        # elements.append(Paragraph("Commentary:", self.styles["Heading2"]))
        # elements.append(Paragraph(" ".join(commentary), self.styles["Normal"]))
        # elements.append(Spacer(1, 0.3 * inch))

        # Create 2x2 grid of charts
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        self._plot_series(
            ax=axs[0, 0],
            series=pd.Series(metrics.equity_curve),
            title="Equity Curve",
            ylabel="Portfolio Value",
        )
        self._plot_series(
            ax=axs[0, 1],
            series=pd.Series(metrics.rolling_drawdown),
            title="Rolling Drawdown",
            ylabel="Drawdown",
        )
        self._plot_series(
            ax=axs[1, 0],
            series=pd.Series(metrics.rolling_sharpe),
            title="Rolling Sharpe Ratio",
            ylabel="Sharpe",
        )
        self._plot_series(
            ax=axs[1, 1],
            series=pd.Series(metrics.daily_returns),
            title="Daily Returns",
            ylabel="Return",
        )

        chart_buf = self._fig_to_buf(fig=fig)
        elements.append(Image(chart_buf, width=6.5 * inch, height=5 * inch))

        overlay_buf = self._plot_trade_overlay(
            market_snapshot=market_snapshot,
            trades=job.equity_curve,
            symbol=job.tickers[0],
        )
        elements.append(Paragraph("Price with Trade Overlay", self.styles["Heading2"]))
        elements.append(Image(overlay_buf, width=8 * inch, height=3.5 * inch))

        buf = self._benchmark_vs_portfolio_performance_figure(
            market_snapshot=market_snapshot, metrics=metrics
        )
        elements.append(Image(buf, width=8 * inch, height=3.5 * inch))

        buf = self._ticker_pnl_performance_figure(job=job)
        elements.append(Image(buf, width=8 * inch, height=3.5 * inch))

        doc.build(elements)
        print(f"Report saved to {filepath}")

    def _benchmark_vs_portfolio_performance_figure(
        self, market_snapshot: MarketSnapshot, metrics: PortfolioMetrics
    ):
        equity_series = pd.Series(metrics.equity_curve)
        portfolio_returns = equity_series.pct_change().fillna(0)
        portfolio_cum = (1 + portfolio_returns).cumprod()

        # Get benchmark price series
        benchmark_series = pd.Series(
            market_snapshot.get(symbol="^SPX", variable="close", with_timestamps=True)
        )
        benchmark_returns = benchmark_series.pct_change().fillna(0)
        benchmark_cum = (1 + benchmark_returns).cumprod()

        # Normalise start date
        common_index = portfolio_cum.index.intersection(benchmark_cum.index)
        portfolio_cum = portfolio_cum.loc[common_index]
        benchmark_cum = benchmark_cum.loc[common_index]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(portfolio_cum, label="Strategy", color="blue")
        ax.plot(benchmark_cum, label="Benchmark (SPX)", color="gray", linestyle="--")
        ax.set_title("Cumulative Returns: Strategy vs Benchmark")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)  # Important to avoid memory leak
        buf.seek(0)
        return buf

    def _ticker_pnl_performance_figure(self, job: StrategyJob):
        trade_data = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "close_date": pd.to_datetime(t.close_date),
                    "pnl": t.pnl,
                }
                for t in job.equity_curve
            ]
        )

        # Group by close_date and symbol, sum PnL (in case multiple trades close same day)
        daily_pnl = (
            trade_data.groupby(["close_date", "symbol"])["pnl"].sum().reset_index()
        )

        # Pivot to have symbols as columns, dates as index, pnl as values
        pnl_pivot = daily_pnl.pivot(
            index="close_date", columns="symbol", values="pnl"
        ).fillna(0)

        # Sort by date index (important)
        pnl_pivot = pnl_pivot.sort_index()

        # Calculate cumulative PnL per symbol
        cumulative_pnl = pnl_pivot.cumsum()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol in cumulative_pnl.columns:
            ax.plot(cumulative_pnl.index, cumulative_pnl[symbol], label=symbol)

        ax.set_title("Cumulative PnL by Symbol")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(loc="upper left", fontsize="small", ncol=3)
        ax.grid(True)
        fig.tight_layout()

        # Save figure to buffer for PDF export
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    def _plot_series(self, ax: Axes, series: pd.Series, title: str, ylabel: str):
        ax.plot(series.index, series.values, label=title, color="steelblue")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Determine date span
        start_date = series.index.min()
        end_date = series.index.max()
        total_years = (end_date - start_date).days / 365.25

        # Adjust x-axis formatting based on date span
        if total_years > 2:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        ax.tick_params(axis="x", rotation=45)

    def _fig_to_buf(self, fig: Figure):
        buf = io.BytesIO()
        fig.tight_layout(pad=4.0)
        fig.savefig(buf, format="PNG", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _plot_trade_overlay(
        self, market_snapshot: MarketSnapshot, trades: list[Trade], symbol: str
    ) -> io.BytesIO:
        return plot_trade_overlay_matplotlib(
            market_snapshot=market_snapshot, trades=trades, symbol=symbol
        )


def plot_trade_overlay_matplotlib(
    market_snapshot: MarketSnapshot,
    trades: list[Trade],
    symbol: str,
) -> io.BytesIO:
    """
    ohlc: DataFrame with columns ['open', 'high', 'low', 'close'] and datetime index
    trades: list of Trade objects with .symbol, .open_date, .close_date, .side, .quantity
    symbol: the ticker for which this chart is generated

    Returns: io.BytesIO with the plot PNG
    """
    open_data = market_snapshot.get(
        symbol=symbol, variable="open", with_timestamps=True
    )
    close_data = market_snapshot.get(symbol=symbol, variable="close")
    low_data = market_snapshot.get(symbol=symbol, variable="low")
    high_data = market_snapshot.get(symbol=symbol, variable="high")

    ohlc_df = build_ohlc_dataframe(open_data, close_data, high_data, low_data)

    df = ohlc_df.copy()
    df.index = pd.to_datetime(df.index)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Format x-axis for date
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Plot candlesticks manually
    width = 0.6
    for _date, row in df.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        fill_color = "none"  # Transparent fill -> black
        ax.plot(
            [_date, _date], [row["low"], row["high"]], color="black", linewidth=1
        )  # wick
        rect = Rectangle(
            (mdates.date2num(_date) - width / 2, min(row["open"], row["close"])),
            width,
            abs(row["close"] - row["open"]),
            facecolor=color,
            edgecolor=fill_color,
            linewidth=1,
        )
        ax.add_patch(rect)

    # Draw trade rectangles + markers
    for trade in trades:
        if trade.symbol != symbol:
            continue

        open_date = pd.to_datetime(trade.open_date)
        close_date = pd.to_datetime(trade.close_date or df.index[-1])

        open_price = df.loc[open_date]["close"]
        close_price = df.loc[close_date]["close"]
        pnl_positive = (close_price - open_price) * trade.quantity > 0

        color = "green" if pnl_positive else "red"

        open_dt = mdates.date2num(pd.to_datetime(trade.open_date))
        close_dt = (
            mdates.date2num(pd.to_datetime(trade.close_date))
            if trade.close_date
            else open_dt + 1
        )
        trade_pnl = (close_price - open_price) * trade.quantity or 0
        pnl_label = f"{trade_pnl:.2f}" if trade_pnl is not None else ""
        alpha = 0.2
        # alpha = min(0.1 + abs(trade.pnl or 0) / 10, 0.8)

        # ax.axvspan(
        #     open_dt,
        #     close_dt,
        #     ymin=0,
        #     ymax=1,  # Full height
        #     facecolor="green" if is_proftrade_pnl > 0 else "red",
        #     alpha=alpha,
        # )

        # # Calculate midpoint x and low y for PnL text
        x_mid = (open_dt + close_dt) / 2
        # y_text = (
        #     ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        # )  # 2% above the bottom

        rect = Rectangle(
            xy=(mdates.date2num(open_date), min(open_price, close_price)),
            width=mdates.date2num(close_date) - mdates.date2num(open_date),
            height=abs(close_price - open_price),
            color=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # # Draw PnL label
        # ax.text(
        #     x_mid,
        #     y_text,
        #     pnl_label,
        #     ha="center",
        #     va="bottom",
        #     fontsize=8,
        #     color="black" if is_profitable else "red",
        #     bbox=dict(
        #         boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7
        #     ),
        # )

        # Calculate center of the box horizontally
        x_mid = (
            mdates.date2num(open_date)
            + (mdates.date2num(close_date) - mdates.date2num(open_date)) / 2
        )

        # Y position just below the box
        y_below_box = min(open_price, close_price) - 0.02 * (
            ax.get_ylim()[1] - ax.get_ylim()[0]
        )

        # Your PnL label
        pnl_label = f"{trade_pnl:.0f}"

        # Add the label
        ax.text(
            x_mid,
            y_below_box,
            pnl_label,
            ha="center",
            va="top",
            fontsize=8,
            color="black" if trade_pnl >= 0 else "red",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7
            ),
            clip_on=True,
        )

        # Markers
        ax.plot(
            open_date,
            open_price,
            "^",
            color="green" if trade.side.lower() == "buy" else "red",
            markersize=8,
        )
        if trade.close_date:
            ax.plot(close_date, close_price, "x", color="black", markersize=8)

    ax.set_title(f"Trade Overlay for {symbol}")
    ax.set_ylabel("Price")
    ax.grid(True)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_ohlc_dataframe(
    open_data: dict, close_data: list, high_data: list, low_data: list
) -> pd.DataFrame:
    # Ensure all are dicts keyed by date
    dates = list(open_data.keys())

    if not (len(dates) == len(close_data) == len(high_data) == len(low_data)):
        raise ValueError("Mismatch in OHLC data lengths")

    close_dict = dict(zip(dates, close_data))
    high_dict = dict(zip(dates, high_data))
    low_dict = dict(zip(dates, low_data))

    # Build DataFrame
    df = pd.DataFrame(
        {
            "open": open_data,
            "high": high_dict,
            "low": low_dict,
            "close": close_dict,
        }
    )

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
