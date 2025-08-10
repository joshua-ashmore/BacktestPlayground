"""Fund Fact Sheet Charts."""

from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

from components.reporter.fund_report.utils.asset_class_map import symbol_asset_class_map
from frontend.db_utils import load_latest_strategy_config


def create_portfolio_chart(df: pd.DataFrame):
    """Create portfolio chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        df["date"],
        df["portfolio_equity_normalized"],
        label="Fund Portfolio",
        linewidth=2,
    )
    ax.grid(True, which="major", axis="y")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    img_buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format="PNG")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def create_strategy_performance_chart(df: pd.DataFrame):
    """Create portfolio chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [col for col in df.columns if "folio" not in col]
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.2 + 0.8 * i / (len(cols) - 1)) for i in range(len(cols))]

    for col, color in zip(cols, colors):
        ax.plot(
            df.index,
            df[col],
            label=col,
            linewidth=2,
            color=color,
        )
    ax.grid(True, which="major", axis="y")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()

    img_buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format="PNG")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def create_monthly_returns(monthly_returns: pd.Series):
    """Create monthly returns bar chart with blue gradient."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Normalize return values for colormap (clip to -10% to +10% to avoid extremes)
    norm = plt.Normalize(
        vmin=min(-0.1, monthly_returns.min()), vmax=max(0.1, monthly_returns.max())
    )
    cmap = plt.get_cmap("Blues")

    # Map each return to a color
    colors = [cmap(norm(val)) for val in monthly_returns.values]

    bars = ax.bar(monthly_returns.index, monthly_returns.values * 100, color=colors)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    img_buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format="PNG", dpi=300)
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def create_asset_split_chart(summary_df: pd.DataFrame):
    """Create asset split chart."""
    all_symbs = []

    for i in [1, 2]:
        strat = summary_df[summary_df["id"] == i].iloc[0]
        strat_config = load_latest_strategy_config(strat["strategy_name"])

        if strat_config.get("regime_engine"):
            for _, reg_config in strat_config["regime_engine"]["strategy_map"].items():
                all_symbs.extend(reg_config["symbols"])
        # TODO: else if strat

    asset_classes = [symbol_asset_class_map.get(sym, "Other") for sym in all_symbs]
    asset_class_counts = Counter(asset_classes)

    labels = list(asset_class_counts.keys())
    sizes = list(asset_class_counts.values())
    fig, ax = plt.subplots(figsize=(8, 4))

    num_colors = len(asset_class_counts)
    cmap = plt.cm.get_cmap("Blues")
    colors = [cmap(0.3 + 0.6 * i / (num_colors - 1)) for i in range(num_colors)]
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
    )
    # ax.pie(sizes, autopct="%1.1f%%", startangle=140)
    # plt.title("Asset Class Exposure Across Strategies")
    ax.axis("equal")
    # ax.legend(labels=labels)

    img_buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format="PNG")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer
