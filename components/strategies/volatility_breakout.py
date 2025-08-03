"""Volatility Breakout Strategy Model."""

from datetime import date
from typing import List, Literal

import numpy as np
import pandas as pd

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg
from components.trades.trade_model import Trade

# class VolatilityBreakoutStrategyJob(Strategy):
#     """Volatility Breakout Strategy Job."""

#     strategy_name: Literal["volatility breakout"] = "volatility breakout"
#     atr_window: int = 14
#     breakout_threshold: float = 1.5

#     def generate_signal_on_date(
#         self, job: StrategyJob, target_date: date, previous_date: date
#     ) -> List[TradeIntent]:
#         """Generate signal on date."""
#         trade_intents: List[TradeIntent] = []
#         volatility_scores: List = []
#         for symbol in [
#             ticker
#             for ticker in job.tickers
#             if ticker in job.market_snapshot.data.keys()
#         ]:
#             atr = self.atr(symbol=symbol, job=job, target_date=previous_date)
#             prices = job.market_snapshot.get(
#                 symbol=symbol, variable="close", max_date=target_date
#             )
#             price = prices[-1]
#             prev_close = prices[-2]
#             if price > prev_close + self.breakout_threshold * atr.iloc[-1]:
#                 signal = "buy"
#             elif price < prev_close - self.breakout_threshold * atr.iloc[-1]:
#                 signal = "sell"
#             else:
#                 continue
#             volatility_scores.append(
#                 {
#                     "symbol": symbol,
#                     "signal": signal,
#                     "date": target_date,
#                     "strategy": self.strategy_name,
#                 }
#             )
#         for score in volatility_scores:
#             trade_intents.append(
#                 TradeIntent(
#                     weight=1 / len(volatility_scores),
#                     **score,
#                 )
#             )
#         return trade_intents

#     def atr(self, symbol: str, job: StrategyJob, target_date: date) -> pd.Series:
#         """
#         Calculate the Average True Range (ATR).

#         Parameters:
#             high (pd.Series): High prices
#             low (pd.Series): Low prices
#             close (pd.Series): Close prices
#             window (int): Rolling window size

#         Returns:
#             pd.Series: ATR values
#         """
#         high: pd.Series = pd.Series(
#             job.market_snapshot.get(
#                 symbol=symbol, variable="high", max_date=target_date
#             )
#         )
#         low: pd.Series = pd.Series(
#             job.market_snapshot.get(symbol=symbol, variable="low", max_date=target_date)
#         )
#         close: pd.Series = pd.Series(
#             job.market_snapshot.get(
#                 symbol=symbol, variable="close", max_date=target_date
#             )
#         )
#         high_low = high - low
#         high_close = (high - close.shift()).abs()
#         low_close = (low - close.shift()).abs()
#         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#         atr_values = true_range.rolling(window=self.atr_window).mean()
#         return atr_values


class VolatilityBreakoutStrategyJob(Strategy):
    """Enhanced Volatility Breakout Strategy Job."""

    strategy_name: Literal["volatility breakout"] = "volatility breakout"
    atr_window: int = 14
    breakout_threshold: float = 1.5
    ma_window: int = 20

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> List[TradeIntent]:
        """Generate signal on date."""
        trade_intents: List[TradeIntent] = []
        volatility_signals: List = []

        for symbol in [
            ticker for ticker in job.tickers if ticker in job.market_snapshot.data
        ]:
            close_prices = pd.Series(
                job.market_snapshot.get(
                    symbol=symbol, variable="close", max_date=previous_date
                )
            )

            if len(close_prices) < max(self.atr_window, self.ma_window) + 2:
                continue  # Not enough data

            # Trend filter
            moving_avg = close_prices.rolling(window=self.ma_window).mean()
            atr = self.atr(symbol=symbol, job=job, target_date=previous_date)

            price = close_prices.iloc[-1]
            prev_close = close_prices.iloc[-2]
            ma_trend = moving_avg.iloc[-2]
            atr_val = atr.iloc[-1]

            if np.isnan(ma_trend) or np.isnan(atr_val):
                continue

            signal = None

            if (
                price > prev_close + self.breakout_threshold * atr_val
                and price > ma_trend
            ):
                signal = "buy"
            elif (
                price < prev_close - self.breakout_threshold * atr_val
                and price < ma_trend
            ):
                signal = "sell"

            if signal:
                volatility_signals.append(
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "date": target_date,
                        "strategy": self.strategy_name,
                    }
                )

        for signal in volatility_signals:
            trade_intents.append(
                TradeIntent(
                    date=signal["date"],
                    legs=[TradeIntentLeg(weight=1 / len(volatility_signals), **signal)],
                )
            )

        return trade_intents

    def atr(self, symbol: str, job: StrategyJob, target_date: date) -> pd.Series:
        """Average True Range (ATR) calculation."""
        high = pd.Series(
            job.market_snapshot.get(
                symbol=symbol, variable="high", max_date=target_date
            )
        )
        low = pd.Series(
            job.market_snapshot.get(symbol=symbol, variable="low", max_date=target_date)
        )
        close = pd.Series(
            job.market_snapshot.get(
                symbol=symbol, variable="close", max_date=target_date
            )
        )

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=self.atr_window).mean()

    def generate_exit_signals(
        self, job: StrategyJob, trade: Trade, current_date: date
    ) -> bool:
        """Generate exit signals for Volatility Breakout trades.

        Exit if price closes below trailing stop (e.g., MA or entry - k * ATR)
        """
        close_prices = pd.Series(
            job.market_snapshot.get(
                symbol=trade.legs[0].symbol, variable="close", max_date=current_date
            )
        )
        if len(close_prices) < self.ma_window + 2:
            return False

        atr_series = self.atr(
            symbol=trade.legs[0].symbol, job=job, target_date=current_date
        )
        moving_avg = close_prices.rolling(window=self.ma_window).mean()

        try:
            current_price = close_prices.iloc[-1]
            atr = atr_series.iloc[-1]
            ma = moving_avg.iloc[-1]
        except IndexError:
            return False

        if np.isnan(current_price) or np.isnan(atr) or np.isnan(ma):
            return False

        exit_signal = False

        if trade.legs[0].side == "buy":
            trailing_stop = min(ma, trade.legs[0].price - 1.0 * atr)
            if current_price < trailing_stop:
                exit_signal = True

        elif trade.legs[0].side == "sell":
            trailing_stop = max(ma, trade.legs[0].price + 1.0 * atr)
            if current_price > trailing_stop:
                exit_signal = True

        return exit_signal


# class VolatilityBreakoutStrategyJob(Strategy):
#     """Enhanced Volatility Breakout Strategy Job with dynamic threshold and filters."""

#     strategy_name: Literal["volatility breakout"] = "volatility breakout"
#     atr_window: int = 14
#     breakout_threshold: float = 1.5
#     ma_window: int = 20  # Trend confirmation window
#     min_avg_volume: float = 1e6  # Minimum average volume filter
#     consolidation_window: int = 10  # Window to check consolidation before breakout

#     def generate_signal_on_date(
#         self, job: StrategyJob, target_date: date, previous_date: date
#     ) -> List[TradeIntent]:
#         trade_intents: List[TradeIntent] = []
#         volatility_signals: List = []

#         for symbol in [
#             ticker for ticker in job.tickers if ticker in job.market_snapshot.data
#         ]:
#             close_prices = pd.Series(
#                 job.market_snapshot.get(
#                     symbol=symbol, variable="close", max_date=previous_date
#                 )
#             )
#             high_prices = pd.Series(
#                 job.market_snapshot.get(
#                     symbol=symbol, variable="high", max_date=previous_date
#                 )
#             )
#             low_prices = pd.Series(
#                 job.market_snapshot.get(
#                     symbol=symbol, variable="low", max_date=previous_date
#                 )
#             )
#             volume = pd.Series(
#                 job.market_snapshot.get(
#                     symbol=symbol, variable="volume", max_date=previous_date
#                 )
#             )

#             if len(close_prices) < max(self.atr_window, self.ma_window) + 2:
#                 continue  # Not enough data

#             # Filter low volume
#             avg_volume = volume[-self.ma_window :].mean()
#             if avg_volume < self.min_avg_volume:
#                 continue

#             # Calculate ATR
#             atr = self.atr(symbol=symbol, job=job, target_date=previous_date)

#             price = close_prices.iloc[-1]
#             prev_close = close_prices.iloc[-2]
#             ma_trend = close_prices.rolling(window=self.ma_window).mean().iloc[-2]
#             atr_val = atr.iloc[-1]

#             if np.isnan(ma_trend) or np.isnan(atr_val):
#                 continue

#             # Check consolidation before breakout (price range low volatility)
#             consolidation_high = high_prices[-self.consolidation_window : -1].max()
#             consolidation_low = low_prices[-self.consolidation_window : -1].min()
#             consolidation_range = consolidation_high - consolidation_low

#             if consolidation_range > 2 * atr_val:
#                 # Price moved too much recently - skip breakout signals to avoid false positives
#                 continue

#             # Dynamic threshold (optional): adjust breakout threshold based on ATR volatility percentile
#             # e.g., breakout_threshold = self.breakout_threshold * (1 + 0.5 * (atr_val / atr.mean()))
#             breakout_threshold = (
#                 self.breakout_threshold
#             )  # keep fixed or implement dynamic here

#             signal = None
#             breakout_size = 0.0

#             if price > prev_close + breakout_threshold * atr_val and price > ma_trend:
#                 signal = "buy"
#                 breakout_size = price - (prev_close + breakout_threshold * atr_val)
#             elif price < prev_close - breakout_threshold * atr_val and price < ma_trend:
#                 signal = "sell"
#                 breakout_size = (prev_close - breakout_threshold * atr_val) - price

#             if signal:
#                 # Weight proportional to breakout strength relative to ATR
#                 weight = breakout_size / atr_val if atr_val != 0 else 1.0
#                 weight = max(min(weight, 1.0), 0.1)  # Clamp weight between 0.1 and 1.0

#                 volatility_signals.append(
#                     {
#                         "symbol": symbol,
#                         "signal": signal,
#                         "date": target_date,
#                         "strategy": self.strategy_name,
#                         "weight": weight,
#                     }
#                 )

#         for signal in volatility_signals:
#             trade_intents.append(TradeIntent(**signal))

#         return trade_intents

#     def atr(self, symbol: str, job: StrategyJob, target_date: date) -> pd.Series:
#         """Average True Range (ATR) calculation."""
#         high = pd.Series(
#             job.market_snapshot.get(
#                 symbol=symbol, variable="high", max_date=target_date
#             )
#         )
#         low = pd.Series(
#             job.market_snapshot.get(symbol=symbol, variable="low", max_date=target_date)
#         )
#         close = pd.Series(
#             job.market_snapshot.get(
#                 symbol=symbol, variable="close", max_date=target_date
#             )
#         )

#         high_low = high - low
#         high_close = (high - close.shift()).abs()
#         low_close = (low - close.shift()).abs()

#         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#         return true_range.rolling(window=self.atr_window).mean()
