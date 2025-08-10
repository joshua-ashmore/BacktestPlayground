"""Tables Util Functions."""

from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle


def generate_performance_table(
    df: pd.DataFrame, equity_columns: List, small_style: ParagraphStyle
):
    """Generate performance table."""
    df["date"] = pd.to_datetime(df.index)
    df.set_index("date", inplace=True)
    ytd_start = pd.date_range(start=f"{df.index[-1].year}-01-01", periods=10, freq="B")
    ytd_start = next(date for date in ytd_start if date in df.index)
    periods = {
        "YTD": ytd_start,
        "1M": df.index[-1] - pd.DateOffset(months=1),
        "3M": df.index[-1] - pd.DateOffset(months=3),
        "6M": df.index[-1] - pd.DateOffset(months=6),
        "1Y": df.index[-1] - pd.DateOffset(years=1),
        "3Y": df.index[-1] - pd.DateOffset(years=3),
        "5Y": df.index[-1] - pd.DateOffset(years=5),
    }

    def calc_return(df, col, start_date, end_date):
        try:
            start_price = df.loc[df.index >= start_date, col].iloc[0]
            end_price = (
                df.loc[end_date, col] if end_date in df.index else df[col].iloc[-1]
            )
            return 100 * ((end_price / start_price) - 1)
        except IndexError:
            return None  # Not enough data

    latest_date = df.index[-1]
    results = []

    for strategy in ["Aggregated Portfolio"] + equity_columns:  # + benchmark
        row = {"Strategy": strategy}
        for label, start_date in periods.items():
            if start_date < df.index[0]:
                ret = None
            else:
                ret = calc_return(df, strategy, start_date, latest_date)
            row[label] = round(ret, 2) if ret is not None else "N/A"
        results.append(row)
    returns_df = pd.DataFrame(results)

    columns = ["Strategy", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"]
    performance_table_data = [columns]

    def format_value(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else v

    for _, row in returns_df.iterrows():
        performance_table_data.append(
            [
                Paragraph(row["Strategy"], small_style),
                format_value(row["YTD"]),
                format_value(row["1M"]),
                format_value(row["3M"]),
                format_value(row["6M"]),
                format_value(row["1Y"]),
                format_value(row["3Y"]),
                format_value(row["5Y"]),
            ]
        )

    col_width_table = 1.4
    performance_table = Table(
        performance_table_data,
        colWidths=[3 * cm] + [col_width_table * cm] * (len(columns) - 1),
    )
    performance_table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )

    return performance_table


def generate_performance_metrics_table(
    summary_df: pd.DataFrame, small_style: ParagraphStyle
):
    """Generate performance metrics table."""
    columns = [
        "Strategy",
        "Ann. Ret.",
        "Vol.",
        "Sharpe",
        "Info. Ratio",
        "Max DD",
        "Win Rate",
        "Turnover",
    ]
    table_data = [columns]
    for _, row in summary_df.iterrows():
        table_data.append(
            [
                Paragraph(row["strategy_name"], small_style),
                f"{row['annualized_return']:.2%}",
                f"{row['annualized_volatility']:.2%}",
                f"{row['sharpe_ratio']:.2f}",
                f"{row['information_ratio']:.2f}",
                f"{row['max_drawdown']:.2%}",
                f"{row['win_rate']:.1%}",
                f"{row['turnover']:.2f}",
            ]
        )

    col_width_table = 1.4
    summary_table = Table(
        table_data,
        colWidths=[3 * cm] + [col_width_table * cm] * (len(columns) - 1),
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )

    return summary_table
