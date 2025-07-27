"""Database."""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.models.records import (
    DailyReturn,
    PriceRecord,
    StrategyMetrics,
    StrategyRun,
)
from components.trades.trade_model import Trade

DB_PATH = Path("data/backtests.db")


class SQLiteDB:
    """SQLite DB Object."""

    def __init__(self, db_path: Path = DB_PATH):
        """Init."""
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def connect(self):
        """Connect to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _ensure_schema(self):
        """Create all tables if they don't exist."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date TEXT,
                close REAL,
                PRIMARY KEY (ticker, date)
            );

            CREATE TABLE IF NOT EXISTS strategy_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                config JSON,
                timestamp TEXT DEFAULT (DATETIME('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_returns (
                run_id INTEGER,
                date TEXT,
                symbol TEXT,
                pnl REAL,
                FOREIGN KEY (run_id) REFERENCES strategy_runs(id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER,
                sharpe REAL,
                total_return REAL,
                max_drawdown REAL,
                FOREIGN KEY (run_id) REFERENCES strategy_runs(id)
            );

            CREATE TABLE IF NOT EXISTS trades (
                symbol TEXT,
                open_date TEXT,
                close_date TEXT,
                quantity REAL,
                price REAL,
                notional REAL,
                strategy TEXT
            );
            """
            )

    def upsert_price(self, record: PriceRecord):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prices (ticker, date, close)
                VALUES (?, ?, ?)
            """,
                (record.ticker, record.date, record.close),
            )

    def insert_strategy_run(self, run: StrategyRun) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO strategy_runs (name, config, timestamp)
                VALUES (?, json(?), ?)
            """,
                (
                    run.name,
                    run.config.model_dump_json(),
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
            return cur.lastrowid

    def insert_daily_returns(self, returns: list[DailyReturn]):
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO daily_returns (run_id, date, symbol, pnl)
                VALUES (:run_id, :date, :symbol, :pnl)
            """,
                [r.model_dump() for r in returns],
            )

    def insert_metrics(self, metrics: StrategyMetrics):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (run_id, sharpe, total_return, max_drawdown)
                VALUES (?, ?, ?, ?)
            """,
                (
                    metrics.run_id,
                    metrics.sharpe,
                    metrics.total_return,
                    metrics.max_drawdown,
                ),
            )

    def get_latest_price_date(self, ticker: str) -> Optional[str]:
        """Get latest price date."""
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT MAX(date) FROM prices WHERE ticker = ?
            """,
                (ticker,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None

    def get_all_strategies(self) -> List[str]:
        """Return all strategies."""
        with self.connect() as conn:
            cur = conn.execute("SELECT name FROM strategy_runs ORDER BY timestamp DESC")
            return [r[0] for r in cur.fetchall()]

    def save_trades_to_sqlite(conn: sqlite3.Connection, trades: List[Trade]):
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO trades (symbol, open_date, close_date, quantity, price, notional, strategy)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    t.symbol,
                    t.open_date.isoformat(),
                    t.close_date.isoformat(),
                    t.quantity,
                    t.price,
                    t.notional,
                    t.strategy,
                )
                for t in trades
            ],
        )
        conn.commit()

    def load_trades_from_sqlite(conn: sqlite3.Connection) -> List[Trade]:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT symbol, date, quantity, price, notional, strategy FROM trades"
        ).fetchall()
        return [
            Trade(
                symbol=row[0],
                date=date.fromisoformat(row[1]),
                quantity=row[2],
                price=row[3],
                notional=row[4],
                strategy=row[5],
            )
            for row in rows
        ]
