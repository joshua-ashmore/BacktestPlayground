"""Metrics Base Model."""

from collections import defaultdict
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from backtester.market_data.market import MarketSnapshot
from components.backtester.base_model import PortfolioSnapshot
from components.trades.trade_model import Trade


class PortfolioMetrics(BaseModel):
    """Portfolio Metrics."""

    start_date: date
    end_date: date
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    rolling_sharpe: dict[date, float]
    rolling_drawdown: dict[date, float]
    equity_curve: dict[date, float]
    daily_returns: dict[date, float]
    num_trades: Optional[int] = None
    win_rate: Optional[float] = None
    average_pnl: Optional[float] = None
    turnover: Optional[float] = None


class MetricsEngine(BaseModel):
    """Metrics Engine."""

    rolling_window: int = 20

    def compute(
        self,
        benchmark_symbol: str,
        market_snapshot: MarketSnapshot,
        snapshots: List[PortfolioSnapshot],
        trades: Optional[List[Trade]] = None,
    ) -> PortfolioMetrics:
        if len(snapshots) < 2:
            raise ValueError("Insufficient portfolio snapshots")

        # Sort snapshots
        snapshots = sorted(snapshots, key=lambda x: x.date)
        dates = [s.date for s in snapshots]
        values = [s.total_value for s in snapshots]
        equity_curve = pd.Series(values, index=pd.to_datetime(dates))

        # Daily returns
        returns = equity_curve.pct_change().dropna()
        cumulative_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

        # Information Ratio
        benchmark_returns = (
            pd.Series(
                market_snapshot.get(
                    symbol=benchmark_symbol,
                    variable="close",
                    dates=dates,
                    with_timestamps=True,
                )
            )
            .pct_change()
            .dropna()
        )
        returns, benchmark_returns = returns.align(benchmark_returns, join="inner")
        active_return = returns - benchmark_returns
        active_return_mean = active_return.mean() * 252
        tracking_error = active_return.std() * (252**0.5)
        information_ratio = active_return_mean / tracking_error

        # Rolling Sharpe
        rolling_sharpe = (
            returns.rolling(self.rolling_window)
            .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
            .dropna()
        )

        # Rolling Max Drawdown
        rolling_max = equity_curve.cummax()
        rolling_drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = rolling_drawdown.min()

        # Trade Metrics
        num_trades = win_rate = average_pnl = None
        if trades:
            for trade in trades:
                if trade.close_date:
                    close_price = market_snapshot.get(
                        symbol=trade.symbol, variable="close", dates=trade.close_date
                    )[0]
                    open_price = market_snapshot.get(
                        symbol=trade.symbol, variable="close", dates=trade.open_date
                    )[0]
                    trade.pnl = (close_price - open_price) * trade.quantity
            pnl_list = [t.pnl for t in trades if t.pnl is not None]
            num_trades = len(pnl_list)
            wins = [p for p in pnl_list if p > 0]
            win_rate = len(wins) / len(pnl_list) if pnl_list else None
            average_pnl = np.mean(pnl_list) if pnl_list else None

        # Turnover (sum of absolute position changes)
        turnover = self._compute_turnover(snapshots=snapshots)

        return PortfolioMetrics(
            start_date=equity_curve.index[0].date(),
            end_date=equity_curve.index[-1].date(),
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            rolling_sharpe={k.date(): v for k, v in rolling_sharpe.items()},
            rolling_drawdown={k.date(): v for k, v in rolling_drawdown.items()},
            equity_curve={k.date(): v for k, v in equity_curve.items()},
            daily_returns={k.date(): v for k, v in returns.items()},
            num_trades=num_trades,
            win_rate=win_rate,
            average_pnl=average_pnl,
            turnover=turnover,
        )

    def _compute_turnover(self, snapshots: List[PortfolioSnapshot]) -> Optional[float]:
        if len(snapshots) < 2:
            return None

        snapshots = sorted(snapshots, key=lambda x: x.date)
        turnover = 0.0
        prev_positions = defaultdict(float)

        for snap in snapshots:
            for sym, qty in snap.positions.items():
                price = snap.prices.get(sym)
                if price is None:
                    continue
                change = abs(qty - prev_positions[sym]) * price
                turnover += change
                prev_positions[sym] = qty

        total_value = snapshots[-1].total_value
        return turnover / total_value if total_value else None
