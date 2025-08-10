"""Angular Momentum Strategy."""

import math
from collections import OrderedDict
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # noqa: F401

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg
from components.trades.trade_model import Trade


class GradientMomentumStrategy(Strategy):
    """Gradient Momentum Strategy with angle + RSI filters."""

    strategy_name: Literal["gradient_momentum"] = "gradient_momentum"
    lookback_window: int = 20
    rsi_window: int = 14
    up_trend_angle_threshold_deg: float = 50
    down_trend_angle_threshold_deg: float = -50
    angle_calculation_day_offset: int = 1
    top_n: int = 3
    max_holding_days: int = 60
    rsi_upper: float = 80
    rsi_lower: float = 20

    def get_required_lookback_days(self) -> int:
        """Longest historical window needed for calculations."""
        return max(
            self.lookback_window,
            self.rsi_window,
            self.angle_calculation_day_offset,
            self.max_holding_days,
        )

    def compute_angle(self, prices: pd.Series) -> float:
        """Compute price trend angle in degrees over the given price series."""
        # Type 1
        if len(prices) < 2:
            return 0.0
        # slope = price change / number of periods
        slope = (prices.iloc[-1] - prices.iloc[0]) / len(prices)
        return math.degrees(np.arctan(slope))

        # Type 2
        # if len(prices) < 2:
        #     return 0.0
        # x = np.arange(len(prices)).reshape(-1, 1)
        # y = prices.values.reshape(-1, 1)
        # model = LinearRegression().fit(x, y)
        # slope = model.coef_[0][0]
        # volatility = prices.pct_change().std() + 1e-10
        # slope_adj = slope / volatility
        # return math.degrees(np.arctan(slope_adj))

        # OG
        # if self.angle_calculation_day_offset > len(prices):
        #     return 0
        # return math.degrees(
        #     np.arctan(
        #         (prices.iloc[-1] - prices.iloc[-self.angle_calculation_day_offset])
        #         / len(prices)
        #     )
        # )

    def compute_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Vectorised RSI calculation."""
        # delta = prices.diff()
        # gain = np.where(delta > 0, delta, 0.0)
        # loss = np.where(delta < 0, -delta, 0.0)

        # avg_gain = pd.Series(gain).rolling(window).mean()
        # avg_loss = pd.Series(loss).rolling(window).mean()

        # rs = avg_gain / (avg_loss + 1e-10)
        # rsi = 100 - (100 / (1 + rs))
        # return float(rsi.iloc[-1])

        # delta = prices.diff()
        # up = delta.clip(lower=0)
        # down = -delta.clip(upper=0)
        # avg_gain = up.ewm(alpha=1 / window, min_periods=window).mean()
        # avg_loss = down.ewm(alpha=1 / window, min_periods=window).mean()
        # rs = avg_gain / (avg_loss + 1e-10)
        # return 100 - (100 / (1 + rs)).iloc[-1]

        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def rank_candidates(
        self, candidates: list[dict], top_n: int, reverse: bool
    ) -> list[dict]:
        """Rank and return top N candidates by trend angle."""
        return sorted(candidates, key=lambda x: x["angle"], reverse=reverse)[:top_n]

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> list[TradeIntent]:

        long_candidates = []
        short_candidates = []

        for symbol in job.tickers:
            if symbol not in job.market_snapshot.data:
                continue

            price_series = job.market_snapshot.get(
                symbol=symbol,
                variable="close",
                max_date=previous_date,
                with_timestamps=True,
            )

            # Ensure chronological order
            prices = pd.Series(OrderedDict(sorted(price_series.items())), dtype=float)
            prices.index = pd.to_datetime(prices.index)

            if len(prices) < max(self.lookback_window, self.rsi_window):
                continue

            angle = self.compute_angle(prices[-self.angle_calculation_day_offset :])
            rsi = self.compute_rsi(prices, window=self.rsi_window)
            current_price = prices.iloc[-1]

            if angle > self.up_trend_angle_threshold_deg and rsi < self.rsi_upper:
                long_candidates.append(
                    {
                        "symbol": symbol,
                        "angle": angle,
                        "rsi": rsi,
                        "entry_price": current_price,
                        "date": target_date,
                        "signal": "buy",
                    }
                )
            elif angle < self.down_trend_angle_threshold_deg and rsi > self.rsi_lower:
                short_candidates.append(
                    {
                        "symbol": symbol,
                        "angle": angle,
                        "rsi": rsi,
                        "entry_price": current_price,
                        "date": target_date,
                        "signal": "sell",
                    }
                )

        # Rank and select top N from each side
        top_longs = self.rank_candidates(long_candidates, self.top_n, reverse=True)
        top_shorts = self.rank_candidates(short_candidates, self.top_n, reverse=False)

        trade_intents = []
        total_trades = len(top_longs) + len(top_shorts)
        weight = 1.0 / total_trades if total_trades > 0 else 0.0

        for entry in top_longs + top_shorts:
            trade_intents.append(
                TradeIntent(
                    date=target_date,
                    legs=[
                        TradeIntentLeg(
                            symbol=entry["symbol"],
                            date=entry["date"],
                            signal=entry["signal"],
                            weight=weight,
                            strategy=self.strategy_name,
                        )
                    ],
                )
            )

        return trade_intents

    def generate_exit_signals(
        self, job: StrategyJob, trade: Trade, current_date: date
    ) -> bool:
        """
        Decide whether to exit a position based on momentum (angle) and RSI conditions.
        Exits can occur if:
        - Max holding period is exceeded
        - Momentum reverses direction
        - RSI hits overbought/oversold extremes
        """

        symbol = trade.legs[0].symbol
        entry_side = trade.legs[0].side  # "buy" or "sell"

        # --- Step 1: Fetch recent prices ---
        price_series = job.market_snapshot.get(
            symbol=symbol,
            variable="close",
            max_date=current_date,
            with_timestamps=True,
        )

        if not price_series:
            return False  # No data available

        prices = pd.Series(OrderedDict(sorted(price_series.items())))
        prices.index = pd.to_datetime(prices.index)

        # Need enough data to calculate indicators
        if current_date not in prices.index or len(prices) < self.lookback_window:
            return False

        # --- Step 2: Check holding period limit ---
        holding_period = (current_date - trade.legs[0].open_date).days
        if holding_period > self.max_holding_days:
            return True

        # --- Step 3: Compute indicators ---
        recent_prices = prices.iloc[-self.lookback_window :]
        angle = self.compute_angle(recent_prices)
        rsi = self.compute_rsi(prices)

        # --- Step 4: Exit logic ---
        if entry_side == "buy":
            # Momentum reversal (downward slope) or overbought RSI
            return angle < 0 or rsi > 75
        elif entry_side == "sell":
            # Momentum reversal (upward slope) or oversold RSI
            return angle > 0 or rsi < 25

        return False
