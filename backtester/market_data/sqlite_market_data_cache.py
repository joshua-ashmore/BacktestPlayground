"""SQLite Market Data Cache."""

import sqlite3
import pandas as pd


class SQLiteMarketCache:
    def __init__(self, db_path: str = "market_data_cache.db"):
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
        return set(df["datetime"].dt.normalize())  # Just keep dates (drop times)

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
