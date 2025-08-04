"""Angular Momentum Strategy."""

import math
from collections import OrderedDict
from datetime import date
from typing import List, Literal

import numpy as np
import pandas as pd

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg
from components.trades.trade_model import Trade


class GradientMomentumStrategy(Strategy):
    """Gradient Momentum Strategy."""

    strategy_name: Literal["gradient_momentum"] = "gradient_momentum"
    lookback_window: int = 20
    rsi_window: int = 14
    up_trend_angle_threshold_deg: float = 50
    down_trend_angle_threshold_deg: float = -50
    angle_calculation_day_offset: int = 1
    top_n: int = 3
    max_holding_days: int = 60

    def compute_angle(self, prices: pd.Series) -> float:
        if self.angle_calculation_day_offset > len(prices):
            return 0
        return math.degrees(
            np.arctan(
                (prices.iloc[-self.angle_calculation_day_offset] - prices.iloc[0])
                / len(prices)
            )
        )

    def compute_rsi(self, prices: pd.Series, window: int = 14) -> float:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> List[TradeIntent]:
        trade_intents: list[TradeIntent] = []
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

            sorted_prices = OrderedDict(sorted(price_series.items()))
            if previous_date not in sorted_prices:
                continue

            prices = pd.Series(sorted_prices)
            prices.index = pd.to_datetime(prices.index)

            if len(prices) < max(self.lookback_window, self.rsi_window):
                continue

            angle = self.compute_angle(prices[-self.lookback_window :])
            rsi = self.compute_rsi(prices, window=self.rsi_window)
            current_price = prices.iloc[-1]

            # Long signal
            if angle > self.up_trend_angle_threshold_deg and rsi < 70:
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

            # Short signal
            elif angle < self.down_trend_angle_threshold_deg and rsi > 30:
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

        # Select top N long/short
        long_ranked = sorted(long_candidates, key=lambda x: x["angle"], reverse=True)
        short_ranked = sorted(short_candidates, key=lambda x: x["angle"])

        top_longs = long_ranked[: self.top_n]
        top_shorts = short_ranked[: self.top_n]

        total_trades = len(top_longs) + len(top_shorts)

        for entry in top_longs + top_shorts:
            weight = 1.0 / total_trades if total_trades > 0 else 0
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
        symbol = trade.legs[0].symbol
        entry_signal = trade.legs[0].side
        price_series = job.market_snapshot.get(
            symbol=symbol,
            variable="close",
            max_date=current_date,
            with_timestamps=True,
        )

        prices = pd.Series(OrderedDict(sorted(price_series.items())))
        prices.index = pd.to_datetime(prices.index)

        if current_date not in prices.index or len(prices) < self.lookback_window:
            return False

        holding_period = (current_date - trade.legs[0].open_date).days
        if holding_period > self.max_holding_days:
            return True

        angle = self.compute_angle(prices[-self.lookback_window :])
        rsi = self.compute_rsi(prices)

        if entry_signal == "buy":
            if angle < 0 or rsi > 75:
                return True
        elif entry_signal == "sell":
            if angle > 0 or rsi < 25:
                return True

        return False
