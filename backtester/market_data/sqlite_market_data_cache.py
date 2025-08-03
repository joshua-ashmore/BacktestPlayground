"""SQLite Market Data Cache."""

import json
import sqlite3
from typing import Dict

import pandas as pd

from components.metrics.base_model import PortfolioMetrics

# from engine.orchestrator import OrchestratorConfig


class SQLiteMarketCache:
    def __init__(self, db_path: str = "data/market_data_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create market data table and index if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    datetime TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, datetime)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol_date ON market_data (symbol, datetime)"
            )

    def get_cached_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, list[dict]]:
        """Return cached data for multiple symbols between start and end date."""
        placeholders = ",".join("?" for _ in symbols)

        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT * FROM market_data
                WHERE symbol IN ({placeholders}) AND datetime BETWEEN ? AND ?
                ORDER BY symbol ASC, datetime ASC
            """
            df = pd.read_sql_query(query, conn, params=[*symbols, start_date, end_date])

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.astype(object)

        grouped = {
            symbol: group.drop(columns=["symbol"]).to_dict(orient="records")
            for symbol, group in df.groupby("symbol")
        }

        return grouped

    def cache_data(self, symbol: str, df: pd.DataFrame):
        """Only insert rows into cache if not already present."""
        if df.empty:
            return

        df["datetime"] = pd.to_datetime(df["datetime"])
        existing_dates = self.get_existing_dates(
            symbol,
            df["datetime"].min().strftime("%Y-%m-%d"),
            (
                df["datetime"].max() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            ).strftime("%Y-%m-%d %H:%M:%S"),
        )

        df["normalized_date"] = df["datetime"].dt.normalize()
        missing_df = df[~df["normalized_date"].isin(existing_dates)].copy()

        if missing_df.empty:
            return

        missing_df["symbol"] = symbol
        missing_df["datetime"] = missing_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        columns = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
        values = missing_df[columns].values.tolist()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO market_data (symbol, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()

    def get_existing_dates(
        self, symbol: str, start_date: str, end_date: str
    ) -> set[pd.Timestamp]:
        """Get a set of datetime values already cached for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT datetime FROM market_data
                WHERE symbol = ? AND datetime BETWEEN ? AND ?
            """
            df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return set(df["datetime"].dt.normalize())

    def is_range_cached(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if the date range is fully cached for the given symbol."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT MIN(datetime) as min_dt, MAX(datetime) as max_dt
                FROM market_data
                WHERE symbol = ?
            """
            result = conn.execute(query, (symbol,)).fetchone()

        if not result or not result[0] or not result[1]:
            return False

        cached_start = pd.to_datetime(result[0])
        cached_end = pd.to_datetime(result[1])

        return cached_start <= pd.to_datetime(start_date) and cached_end + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        ) >= pd.to_datetime(end_date)


class SQLiteMetricsStore:
    def __init__(self, db_path: str = "data/portfolio_metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    cumulative_return REAL,
                    annualized_return REAL,
                    annualized_volatility REAL,
                    sharpe_ratio REAL,
                    information_ratio REAL,
                    max_drawdown REAL,
                    num_trades INTEGER,
                    win_rate REAL,
                    average_pnl REAL,
                    turnover REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_timeseries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    equity_curve REAL,
                    daily_return REAL,
                    rolling_sharpe REAL,
                    rolling_drawdown REAL,
                    regime_timeseries REAL,
                    FOREIGN KEY (summary_id) REFERENCES portfolio_summary(id)
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_strategy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id INTEGER NOT NULL,
                    regime TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    FOREIGN KEY(summary_id) REFERENCES portfolio_summary(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_strategy_config(self, strategy_name: str, config):
        config_json = json.dumps(config.model_dump(), default=str, indent=2)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_config (strategy_name, config_json)
                VALUES (?, ?)
                """,
                (strategy_name, config_json),
            )

    def load_latest_strategy_config(self, strategy_name: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
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

    def save_portfolio_metrics(self, metrics: PortfolioMetrics, strategy_name: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Insert summary
        c.execute(
            """
            INSERT INTO portfolio_summary (
                strategy_name, start_date, end_date, cumulative_return, annualized_return,
                annualized_volatility, sharpe_ratio, information_ratio,
                max_drawdown, num_trades, win_rate, average_pnl, turnover
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_name,
                metrics.start_date,
                metrics.end_date,
                metrics.cumulative_return,
                metrics.annualized_return,
                metrics.annualized_volatility,
                metrics.sharpe_ratio,
                metrics.information_ratio,
                metrics.max_drawdown,
                metrics.num_trades,
                metrics.win_rate,
                metrics.average_pnl,
                metrics.turnover,
            ),
        )
        summary_id = c.lastrowid

        # Insert timeseries
        for d in metrics.equity_curve:
            c.execute(
                """
                INSERT INTO portfolio_timeseries (
                    summary_id, date, equity_curve, daily_return,
                    rolling_sharpe, rolling_drawdown, regime_timeseries
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    d,
                    metrics.equity_curve.get(d),
                    metrics.daily_returns.get(d),
                    metrics.rolling_sharpe.get(d),
                    metrics.rolling_drawdown.get(d),
                    metrics.regime_timeseries.get(d),
                ),
            )

        for regime, metric_dict in metrics.strategy_metrics.items():
            for metric_name, value in metric_dict.items():
                c.execute(
                    """
                    INSERT INTO portfolio_strategy_metrics (
                        summary_id, regime, metric_name, metric_value
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (summary_id, regime, metric_name, float(value)),
                )

        conn.commit()
        conn.close()
        return summary_id

    def load_timeseries(self, summary_id: int) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT date, equity_curve, daily_return, rolling_sharpe, rolling_drawdown, regime_drawdown
            FROM portfolio_timeseries
            WHERE summary_id = ?
            ORDER BY date ASC
            """,
            conn,
            params=(summary_id,),
            parse_dates=["date"],
        )
        conn.close()
        return df

    def load_strategy_metrics(self, summary_id: int) -> Dict[str, Dict[str, float]]:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT regime, metric_name, metric_value
            FROM portfolio_strategy_metrics
            WHERE summary_id = ?
            """,
            conn,
            params=(summary_id,),
        )
        conn.close()

        strat_dict = {}
        for _, row in df.iterrows():
            strat = row["strategy"]
            if strat not in strat_dict:
                strat_dict[strat] = {}
            strat_dict[strat][row["metric_name"]] = row["metric_value"]
        return strat_dict

    def load_summary(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM portfolio_summary ORDER BY created_at DESC", conn
        )
        conn.close()
        return df
