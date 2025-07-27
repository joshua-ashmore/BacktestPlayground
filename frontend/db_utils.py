"""Util Database Functions."""

import json
import sqlite3

import pandas as pd

from components.metrics.base_model import PortfolioMetrics

DB_PATH = "portfolio_metrics.db"


def get_summary_df():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM portfolio_summary ORDER BY created_at DESC",
            conn,
            parse_dates=["created_at"],
        )


def get_timeseries_df(summary_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT date, equity_curve, daily_return, rolling_sharpe, rolling_drawdown
            FROM portfolio_timeseries
            WHERE summary_id = ?
            ORDER BY date ASC
            """,
            conn,
            params=(summary_id,),
            parse_dates=["date"],
        )


def load_latest_strategy_config(strategy_name: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT config_json FROM strategy_config
            WHERE strategy_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (strategy_name,),
        ).fetchone()

        if row:
            return json.loads(row[0])
        return None


def metrics_to_dataframe(metrics: PortfolioMetrics) -> pd.DataFrame:
    equity_curve = pd.Series(metrics.equity_curve, name="equity_curve")
    daily_returns = pd.Series(metrics.daily_returns, name="daily_return")
    rolling_sharpe = pd.Series(metrics.rolling_sharpe, name="rolling_sharpe")
    rolling_drawdown = pd.Series(metrics.rolling_drawdown, name="rolling_drawdown")
    df = pd.concat(
        [equity_curve, daily_returns, rolling_sharpe, rolling_drawdown], axis=1
    )
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)
    return df


def orchestrator_config_to_df_simple(metrics: PortfolioMetrics) -> pd.DataFrame:
    data = {
        "start_date": metrics.start_date,
        "end_date": metrics.end_date,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "information_ratio": metrics.information_ratio,
        "max_drawdown": metrics.max_drawdown,
        "num_trades": metrics.num_trades,
        "win_rate": metrics.win_rate,
        "average_pnl": metrics.average_pnl,
        "turnover": metrics.turnover,
    }

    df = pd.DataFrame([data])
    return df
