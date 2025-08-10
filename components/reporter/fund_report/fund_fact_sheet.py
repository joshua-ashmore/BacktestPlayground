"""Automated Fund Fact Sheet."""

from datetime import datetime
from functools import reduce

import pandas as pd
from numpy import average
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from components.reporter.fund_report.utils.charts import (
    create_asset_split_chart,
    create_monthly_returns,
    create_portfolio_chart,
    create_strategy_performance_chart,
)
from components.reporter.fund_report.utils.tables_util import (
    generate_performance_metrics_table,
    generate_performance_table,
)
from frontend.db_utils import get_summary_df, get_timeseries_df

# Data
summary_df = get_summary_df()

# Page settings
styles = getSampleStyleSheet()
width, height = A4
margin = 0.5 * cm

col_width = (width - 2 * margin) / 3
row_height = (height - 2 * margin - 2 * cm) / 3
header_height = 2 * cm

# Styles
small_style = ParagraphStyle(
    name="SmallNormal", parent=styles["Normal"], fontSize=8, leading=8
)

frames = [
    Frame(
        margin,
        height - margin - header_height,
        width - 2 * margin,
        header_height,
        id="F1",
    ),
    Frame(
        margin,
        height - margin - header_height - row_height,
        col_width,
        row_height,
        id="F2",
    ),
    Frame(
        margin + col_width,
        height - margin - header_height - row_height,
        2 * col_width,
        row_height,
        id="F3",
    ),
    Frame(
        margin,
        height - margin - header_height - 2 * row_height,
        col_width,
        row_height,
        id="F4",
    ),
    Frame(
        margin + col_width,
        height - margin - header_height - 2 * row_height,
        2 * col_width,
        row_height,
        id="F5",
    ),
    Frame(
        margin,
        height - margin - header_height - 3 * row_height,
        col_width,
        row_height,
        id="F6",
    ),
    Frame(
        margin + col_width,
        height - margin - header_height - 3 * row_height,
        2 * col_width,
        row_height,
        id="F7",
    ),
]

# --- Chart --- #
# Prepare data for chart
strategy_ids = [i for i in range(1, len(summary_df) + 1)]
equity_dfs = []

for sid in strategy_ids:
    df = get_timeseries_df(sid)[["date", "equity_curve"]].copy()
    name = summary_df.loc[summary_df["id"] == sid, "strategy_name"].values[0]
    df.rename(columns={"equity_curve": f"{name}"}, inplace=True)
    equity_dfs.append(df)

combined_df = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"), equity_dfs
)
combined_df.sort_values("date", inplace=True)
combined_df.reset_index(drop=True, inplace=True)
combined_df.fillna(method="ffill", inplace=True)
combined_df.fillna(method="bfill", inplace=True)
equity_columns = [col for col in combined_df.columns if not col.startswith("date")]
combined_df["Aggregated Portfolio"] = combined_df[equity_columns].mean(axis=1)
combined_df["portfolio_equity_normalized"] = (
    combined_df["Aggregated Portfolio"]
    / combined_df["Aggregated Portfolio"].iloc[0]
    * 100
)

# Create the chart image flowable
image_stream = create_portfolio_chart(combined_df)
scale = 0.4
chart_img = Image(image_stream, width=scale * 6 * inch, height=scale * 3 * inch)

# --- Secondary Chart ---- ####
df = combined_df
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
monthly_returns = df["Aggregated Portfolio"].resample("ME").last().pct_change().dropna()
monthly_returns.index = monthly_returns.index.strftime("%b")
# Create the chart image flowable
monthly_image_stream = create_monthly_returns(monthly_returns=monthly_returns)
scale = 0.4
monthly_image_stream = Image(
    monthly_image_stream, width=scale * 6 * inch, height=scale * 3 * inch
)


# Story with placeholders for each frame
story = []

# Frame F1 - Top full width header
today_date_string = datetime.now().date().strftime("%d %b %Y")
story.append(
    Paragraph("Ashmore Multi-Strategy Fund - Monthly Report", styles["Heading2"])
)
story.append(Paragraph(f"{today_date_string}", styles["Normal"]))
story.append(FrameBreak())

# Frame F2 - Left column, row 1
fund_details = [
    [
        Paragraph("Fund Name", small_style),
        Paragraph("Ashmore Multi-Strategy", small_style),
    ],
    [Paragraph("AUM", small_style), Paragraph("£300,000", small_style)],
    [Paragraph("Benchmark", small_style), Paragraph("SPY", small_style)],
    [Paragraph("Inception Date", small_style), Paragraph("2025-01-03", small_style)],
    [Paragraph("Manager", small_style), Paragraph("Joshua Ashmore", small_style)],
]

# Fund Name + Strategy Description
# AUM / Notional Value
# Benchmark (if any)
# Inception Date
# Manager / PM Info
# Monthly Commentary (brief 2–3 lines)

table = Table(fund_details, colWidths=[2 * cm, 4 * cm])
table.setStyle(
    TableStyle(
        [
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.black),
        ]
    )
)
story.append(HRFlowable(width="100%", thickness=2, color=colors.ReportLabFidRed))
story.append(Paragraph("Fund Details", styles["Heading5"]))
# story.append(Spacer(1, 0.2 * cm))
story.append(table)
# story.append(Spacer(1, 0.5 * cm))

# ----- Generate commentary -----
report_month = pd.to_datetime(summary_df["end_date"].iloc[0]).strftime("%B %Y")


def generate_pm_commentary(summary_df, month_name="August", ytd_label="Year-to-date"):
    """
    Generate a PM-style commentary from a summary DataFrame.
    Expects columns: 'Strategy', 'Return', 'Win Rate', 'Trades', 'Avg PnL/Trade', 'Max Drawdown'
    """
    import pandas as pd

    # Ensure numeric sorting works
    df = summary_df.copy()
    for col in [
        "cumulative_return",
        "win_rate",
        "num_trades",
        "average_pnl",
        "max_drawdown",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Identify best and worst performers
    best = df.loc[df["cumulative_return"].idxmax()]
    worst = df.loc[df["cumulative_return"].idxmin()]

    # Create narrative
    commentary = []

    # Intro
    commentary.append(
        f"The multi-strategy portfolio delivered a <b>{average(df['cumulative_return']):.1%}</b> return {ytd_label.lower()}, "  # noqa: E501
        f"with notable strength in <b>{best['strategy_name']}</b>, which gained <b>{best['cumulative_return']:.1%}</b>."
    )

    # Best performer details
    commentary.append(
        f"The {best['strategy_name']} strategy benefited from well-timed market entries, "
        f"achieving a <b>{best['win_rate']:.1%} win rate</b> over {int(best['num_trades'])} trades, "
        # f"with an average profit per trade of <b>£{best['average_pnl']:.0f}</b>, "
        f"while maintaining controlled drawdowns of <b>{best['max_drawdown']:.1%}</b>."
        f"<br/>"
        f"<br/>"
    )

    # Worst performer details
    commentary.append(
        f"In contrast, <b>{worst['strategy_name']}</b> underperformed, returning <b>{worst['cumulative_return']:.1%}</b>, "
        f"due to challenging market conditions and increased noise during this period."
        f"<br/>"
        f"<br/>"
    )

    # Portfolio risk and volatility
    commentary.append(
        "Portfolio volatility remained within expected ranges, "
        "and drawdowns were contained across strategies, reflecting disciplined risk management."
    )

    # Forward-looking outlook
    # commentary.append(
    #     f"Looking ahead into {month_name}, our regime overlay signals a tilt toward a <b>trending market environment</b>, "
    #     "which should favor momentum and breakout strategies. Nonetheless, elevated macroeconomic uncertainty "
    #     "warrants a balanced allocation and selective positioning in high-conviction trades."
    # )

    return "\n\n".join(commentary)


# fund_return = summary_df["cumulative_return"].mean()
# benchmark_return = 0.15  # placeholder benchmark
# outperformance = fund_return - benchmark_return

# # Top strategies
# top_strat = summary_df.loc[summary_df["cumulative_return"].idxmax()]
# second_strat = summary_df.loc[summary_df["cumulative_return"].nlargest(2).index[1]]

# # Regime outlook (pick from regime detection strategies if they exist)
# regime_strats = summary_df[summary_df["strategy_name"].str.contains("Regime Detection")]
# if not regime_strats.empty:
#     latest_regime_strat = regime_strats.iloc[0]
#     regime_profile = (
#         "trending" if latest_regime_strat["sharpe_ratio"] > 1 else "volatile"
#     )
# else:
#     regime_profile = "balanced"

# # Commentary text
# p_summary = (
#     f"The fund returned {fund_return:.2%} in {report_month}, "
#     f"outperforming its benchmark by {outperformance:.2%}."
# )
# d_attribution = (
#     f"Gains were led by the {top_strat['strategy_name']} "
#     f"({top_strat['cumulative_return']:.2%}) and "
#     f"{second_strat['strategy_name']} ({second_strat['cumulative_return']:.2%}) strategies."
# )
# f_outlook = (
#     f"The multi-regime overlay rotated into a {regime_profile} profile "
#     f"by the end of {report_month}."
# )
story.append(Paragraph("Monthly Commentary", styles["Heading5"]))
commentary = generate_pm_commentary(
    summary_df=summary_df,
    month_name=pd.to_datetime(summary_df["end_date"].iloc[0]).strftime("%B"),
)
story.append(Paragraph(f"{commentary}", small_style))
# story.append(Paragraph(f"{p_summary} {d_attribution} {f_outlook}", small_style))


# Performance Summary
# "The fund returned 2.6% in July, outperforming its benchmark by 0.8%."
# Driver Attribution
# "Gains were led by the trend-following and volatility breakout strategies."
# Market Context (if relevant)
# "Markets rallied broadly following dovish Fed comments."
# Forward Outlook or Observations
# "The multi-regime overlay rotated into a risk-on profile by mid-month."

story.append(FrameBreak())

# Frame F3 - Right column, row 1 (2/3 width)
story.append(HRFlowable(width="100%", thickness=2, color=colors.ReportLabFidRed))
story.append(Paragraph("Strategy Performance", styles["Heading5"]))
story.append(Spacer(1, 0.2 * cm))
strat_chart_stream = create_strategy_performance_chart(combined_df)
scale = 0.8
strategy_chart_img = Image(
    strat_chart_stream, width=scale * 6 * inch, height=scale * 3 * inch
)
story.append(strategy_chart_img)
story.append(FrameBreak())

# Frame F4 - Left column, row 2
story.append(HRFlowable(width="100%", thickness=2, color=colors.blue))
story.append(Paragraph("Fund Performance (rebased to 100)", styles["Heading5"]))
story.append(chart_img)
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("Monthly Returns", styles["Heading5"]))
story.append(monthly_image_stream)
story.append(FrameBreak())

# Frame F5 - Right column, row 2
performance_table = generate_performance_table(
    df=combined_df, equity_columns=equity_columns, small_style=small_style
)
story.append(HRFlowable(width="100%", thickness=2, color=colors.blue))
story.append(Paragraph("Aggregate Performance (%)", styles["Heading5"]))
story.append(performance_table)
story.append(FrameBreak())

# Frame F6 - Left column, row 3
story.append(HRFlowable(width="100%", thickness=2, color=colors.blue))
story.append(Paragraph("Fund Risk", styles["Heading5"]))

# Vol breakdown?
# Exposure by asset class

asset_split_image_stream = create_asset_split_chart(summary_df=summary_df)
scale = 0.4
asset_split_image_stream = Image(
    asset_split_image_stream, width=scale * 6 * inch, height=scale * 3 * inch
)
story.append(asset_split_image_stream)

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Regime Timeline", styles["Heading5"]))

# Correlation matrix
story.append(FrameBreak())


# Frame F7 - Right column, row 3
summary_table = generate_performance_metrics_table(
    summary_df=summary_df, small_style=small_style
)
story.append(HRFlowable(width="100%", thickness=2, color=colors.blue))
story.append(Paragraph("Strategy Performance Metrics", styles["Heading5"]))
story.append(summary_table)

# Now you can build your doc using the frames and this story
doc = BaseDocTemplate("output.pdf", pagesize=A4)
template = PageTemplate(id="mypage", frames=frames)
doc.addPageTemplates([template])

doc.build(story)
