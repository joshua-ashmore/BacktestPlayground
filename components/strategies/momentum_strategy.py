"""Momentum Strategy."""

from datetime import date
from typing import Dict, List, Literal, OrderedDict

import numpy as np
import pandas as pd

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg
from components.trades.trade_model import Trade


# class MomentumStrategyJob(Strategy):
#     """Momentum Strategy Job."""

#     strategy_name: Literal["momentum"] = "momentum"
#     base_window: int = 20
#     top_n: int = 3

#     def generate_signal_on_date(
#         self, job: StrategyJob, target_date: date, previous_date: date
#     ) -> List[TradeIntent]:
#         """Generate signals for all provided dates (or current_date if none given)."""
#         trade_intents: list[TradeIntent] = []
#         momentum_scores: List = []

#         for symbol in job.tickers:
#             if symbol not in job.market_snapshot.data.keys():
#                 continue
#             price_series: Dict[str, float] = job.market_snapshot.get(
#                 symbol=symbol,
#                 variable="close",
#                 max_date=previous_date,
#                 with_timestamps=True,
#             )

#             # Ensure sorted by date
#             sorted_prices = OrderedDict(sorted(price_series.items()))
#             date_list = list(sorted_prices.keys())

#             prices = pd.Series(sorted_prices)
#             returns: pd.Series = np.log(prices).diff()

#             realized_vol = returns.rolling(self.base_window).std()
#             vol_percentile = (realized_vol.rank(pct=True)).iloc[-1]
#             if np.isnan(vol_percentile):
#                 momentum_window = 60
#             else:
#                 momentum_window = int(np.clip(90 - 70 * vol_percentile, 20, 90))

#             if previous_date not in date_list:
#                 continue

#             idx = date_list.index(previous_date)
#             if idx < momentum_window:
#                 continue

#             try:
#                 past_date = date_list[idx - momentum_window]
#                 past_price = sorted_prices[past_date]
#                 current_price = self.get_previous_price(
#                     sorted_prices=sorted_prices, target_date=previous_date
#                 )

#                 if past_price is None or current_price is None:
#                     continue

#                 momentum = current_price / past_price - 1
#                 momentum_scores.append(
#                     {"symbol": symbol, "momentum": momentum, "date": target_date}
#                 )

#             except (IndexError, KeyError):
#                 continue

#         # Rank and assign weights
#         if not momentum_scores:
#             return []

#         ranked = sorted(momentum_scores, key=lambda x: x["momentum"], reverse=True)
#         n = len(ranked)
#         top = min(self.top_n, n)

#         for i, symbol in enumerate(ranked):
#             momentum_value = symbol["momentum"]

#             if i < top and momentum_value > 0:
#                 signal = "buy"
#                 weight = 1.0 / top
#             elif i >= n - top and momentum_value < 0:
#                 signal = "sell"
#                 weight = 1.0 / top
#             else:
#                 signal = "hold"
#                 weight = 0.0

#             trade_intents.append(
#                 TradeIntent(
#                     symbol=symbol["symbol"],
#                     date=symbol["date"],
#                     signal=signal,
#                     weight=weight,
#                     strategy=self.strategy_name,
#                 )
#             )

#         return trade_intents


# class MomentumStrategyJob(Strategy):
#     """Momentum Strategy Job with volatility adjustment and filters."""

#     strategy_name: Literal["momentum"] = "momentum"
# base_window: int = Field(
#     default=20, description="Window to calculate realised volatility on"
# )
# top_n: int = Field(
#     default=3, description="Number of symbols in each bucket (buy/sell)"
# )
# min_momentum_threshold: float = Field(
#     default=0.01, description="1% minimum momentum for trade signals"
# )
# min_avg_volume: float = Field(
#     default=1e6, description="Example minimum average volume filter"
# )

#     def generate_signal_on_date(
#         self, job: StrategyJob, target_date: date, previous_date: date
#     ) -> List[TradeIntent]:
#         trade_intents: list[TradeIntent] = []
#         momentum_scores: List = []

#         for symbol in job.tickers:
#             if symbol not in job.market_snapshot.data.keys():
#                 continue

#             # Fetch closing prices with timestamps
#             price_series: Dict[date, float] = job.market_snapshot.get(
#                 symbol=symbol,
#                 variable="close",
#                 max_date=previous_date,
#                 with_timestamps=True,
#             )

#             # Optional: filter low volume symbols
#             volume_series: Dict[date, float] = job.market_snapshot.get(
#                 symbol=symbol,
#                 variable="volume",
#                 max_date=previous_date,
#                 with_timestamps=True,
#             )
#             avg_volume = np.mean(list(volume_series.values())) if volume_series else 0
#             if avg_volume < self.min_avg_volume:
#                 continue

#             # Prepare sorted price series
#             sorted_prices = OrderedDict(sorted(price_series.items()))
#             date_list = list(sorted_prices.keys())

#             if previous_date not in date_list:
#                 continue

#             prices = pd.Series(sorted_prices)
#             returns: pd.Series = np.log(prices).diff()

#             # Calculate realized volatility and dynamic momentum window
#             realized_vol = returns.rolling(self.base_window).std()
#             vol_percentile = (
#                 realized_vol.rank(pct=True).iloc[-1]
#                 if not realized_vol.empty
#                 else np.nan
#             )
#             momentum_window = (
#                 int(np.clip(90 - 70 * vol_percentile, 20, 90))
#                 if not np.isnan(vol_percentile)
#                 else 60
#             )

#             idx = date_list.index(previous_date)
#             if idx < momentum_window:
#                 continue

#             try:
#                 past_date = date_list[idx - momentum_window]
#                 past_price = sorted_prices[past_date]
#                 current_price = sorted_prices[previous_date]

#                 if past_price is None or current_price is None:
#                     continue

#                 # Calculate momentum return
#                 momentum = current_price / past_price - 1

#                 # Momentum threshold filter
#                 if abs(momentum) < self.min_momentum_threshold:
#                     continue

#                 # Secondary confirmation: simple moving average crossover
#                 sma_short = prices.rolling(window=5).mean()
#                 sma_long = prices.rolling(window=momentum_window).mean()
#                 if pd.isna(sma_short[previous_date]) or pd.isna(
#                     sma_long[previous_date]
#                 ):
#                     continue
#                 if momentum > 0 and sma_short[previous_date] <= sma_long[previous_date]:
#                     continue  # No confirmation for long signal
#                 if momentum < 0 and sma_short[previous_date] >= sma_long[previous_date]:
#                     continue  # No confirmation for short signal

#                 momentum_scores.append(
#                     {"symbol": symbol, "momentum": momentum, "date": target_date}
#                 )

#             except (IndexError, KeyError):
#                 continue

#         if not momentum_scores:
#             return []

#         # Sort descending by momentum
#         ranked = sorted(momentum_scores, key=lambda x: x["momentum"], reverse=True)
#         n = len(ranked)
#         top = min(self.top_n, n)

#         # Calculate weights proportional to momentum magnitude in top and bottom ranks
#         top_momentum_sum = sum(abs(x["momentum"]) for x in ranked[:top]) or 1
#         bottom_momentum_sum = sum(abs(x["momentum"]) for x in ranked[-top:]) or 1

#         for i, symbol_data in enumerate(ranked):
#             momentum_value = symbol_data["momentum"]
#             weight = 0.0
#             signal = "hold"

#             if i < top and momentum_value > 0:
#                 signal = "buy"
#                 weight = abs(momentum_value) / top_momentum_sum
#             elif i >= n - top and momentum_value < 0:
#                 signal = "sell"
#                 weight = abs(momentum_value) / bottom_momentum_sum

#             trade_intents.append(
#                 TradeIntent(
#                     symbol=symbol_data["symbol"],
#                     date=symbol_data["date"],
#                     signal=signal,
#                     weight=weight,
#                     strategy=self.strategy_name,
#                 )
#             )

#         return trade_intents


# class MomentumStrategyJob(Strategy):
#     """Enhanced Momentum Strategy Job with filters and exits."""

#     strategy_name: Literal["momentum"] = "momentum"
#     base_window: int = 20
#     top_n: int = 3
#     momentum_threshold: float = 0.02
#     ma_window: int = 50
#     max_holding_days: int = 60

#     def generate_signal_on_date(
#         self, job: StrategyJob, target_date: date, previous_date: date
#     ) -> List[TradeIntent]:
#         trade_intents: list[TradeIntent] = []
#         momentum_scores: List = []

#         for symbol in job.tickers:
#             if symbol not in job.market_snapshot.data:
#                 continue

#             price_series: Dict[str, float] = job.market_snapshot.get(
#                 symbol=symbol,
#                 variable="close",
#                 max_date=previous_date,
#                 with_timestamps=True,
#             )

#             # Sort and validate
#             sorted_prices = OrderedDict(sorted(price_series.items()))
#             if previous_date not in sorted_prices:
#                 continue

#             prices = pd.Series(sorted_prices)
#             prices.index = pd.to_datetime(prices.index)
#             date_list = list([ind.date() for ind in prices.index])

#             # Compute returns
#             returns: pd.Series = np.log(prices).diff().dropna()
#             if len(returns) < self.base_window:
#                 continue

#             # Adaptive momentum window
#             vol = returns.rolling(self.base_window).std()
#             vol_pct = vol.rank(pct=True).iloc[-1]
#             momentum_window = int(np.clip(90 - 70 * vol_pct, 20, 90))

#             if date_list.index(previous_date) < momentum_window:
#                 continue

#             past_price = prices.iloc[date_list.index(previous_date) - momentum_window]
#             current_price = prices.iloc[-1]
#             momentum = current_price / past_price - 1

#             # Trend filter: price must be above MA
#             if len(prices) < self.ma_window:
#                 continue
#             moving_avg = prices.rolling(self.ma_window).mean().iloc[-1]
#             if current_price < moving_avg:
#                 continue

#             if momentum > self.momentum_threshold:
#                 momentum_scores.append(
#                     {
#                         "symbol": symbol,
#                         "momentum": momentum,
#                         "date": target_date,
#                         "entry_price": current_price,
#                     }
#                 )

#         if not momentum_scores:
#             return []

#         # Rank and select top_n
#         ranked = sorted(momentum_scores, key=lambda x: x["momentum"], reverse=True)
#         top = ranked[: min(self.top_n, len(ranked))]

#         for entry in top:
#             trade_intents.append(
#                 TradeIntent(
#                     date=entry["date"],
#                     legs=[
#                         TradeIntentLeg(
#                             symbol=entry["symbol"],
#                             date=entry["date"],
#                             signal="buy",
#                             weight=1.0 / len(top),
#                             strategy=self.strategy_name,
#                         )
#                     ],
#                 )
#             )

#         return trade_intents

#     def generate_exit_signals(
#         self, job: StrategyJob, trade: Trade, current_date: date
#     ) -> bool:
#         """
#         Exit condition:
#         - Momentum reversal
#         - Price falls below moving average
#         - Max holding period exceeded
#         """
#         symbol = trade.legs[0].symbol
#         price_series = job.market_snapshot.get(
#             symbol=symbol,
#             variable="close",
#             max_date=current_date,
#             with_timestamps=True,
#         )

#         prices = pd.Series(OrderedDict(sorted(price_series.items())))
#         prices.index = pd.to_datetime(prices.index)

#         if current_date not in prices.index:
#             return False

#         current_price = prices.loc[current_date]
#         ma = prices.rolling(window=self.ma_window).mean().loc[current_date]

#         # Check holding period
#         holding_period = (current_date - trade.legs[0].open_date).days
#         if holding_period > self.max_holding_days:
#             return True

#         # Momentum reversal
#         lookback_window = min(20, len(prices) - 1)
#         if lookback_window >= 2:
#             momentum = current_price / prices.iloc[-lookback_window] - 1
#             if momentum < 0:
#                 return True

#         # Trend break
#         if current_price < ma:
#             return True

#         return False


class MomentumStrategyJob(Strategy):
    """Enhanced Momentum Strategy with risk-adjusted scoring, trend filtering, and softmax weights."""

    strategy_name: Literal["momentum"] = "momentum"
    base_window: int = 20
    top_n: int = 3
    momentum_threshold: float = 0.02
    ma_window: int = 50
    max_holding_days: int = 60
    min_atr_window: int = 14
    decay_limit: int = 3  # Max number of recent signals allowed for same symbol

    def compute_atr(self, prices: pd.Series, window: int = 14) -> float:
        return prices.diff().abs().rolling(window).mean().iloc[-1]

    def compute_risk_adjusted_momentum(self, prices: pd.Series, window: int) -> float:
        returns = np.log(prices).diff().dropna()
        if len(returns) < window:
            return np.nan
        past_price = prices.iloc[-window]
        current_price = prices.iloc[-1]
        momentum = current_price / past_price - 1
        volatility = returns[-window:].std()
        return momentum / (volatility + 1e-6)

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> List[TradeIntent]:
        trade_intents: list[TradeIntent] = []
        scored_assets: list[Dict] = []

        for symbol in job.tickers:
            if symbol not in job.market_snapshot.data:
                continue

            price_series: Dict[str, float] = job.market_snapshot.get(
                symbol=symbol,
                variable="close",
                max_date=previous_date,
                with_timestamps=True,
            )

            prices = pd.Series(OrderedDict(sorted(price_series.items())))
            prices.index = pd.to_datetime(prices.index)

            if len(prices) < max(self.base_window, self.ma_window, self.min_atr_window):
                continue

            date_list = [d.date() for d in prices.index]
            if previous_date not in date_list:
                continue

            idx = date_list.index(previous_date)

            # Adaptive momentum window
            returns = np.log(prices).diff().dropna()
            vol = returns.rolling(self.base_window).std()
            vol_pct = vol.rank(pct=True).iloc[-1]
            momentum_window = int(np.clip(90 - 70 * vol_pct, 20, 90))

            if idx < momentum_window:
                continue

            risk_adj_mom = self.compute_risk_adjusted_momentum(prices, momentum_window)
            if np.isnan(risk_adj_mom) or risk_adj_mom < self.momentum_threshold:
                continue

            # Trend filter
            ma = prices.rolling(self.ma_window).mean().iloc[-1]
            current_price = prices.iloc[-1]
            if current_price < ma:
                continue

            # Overheating check (momentum decay)
            # signal_count = job.signal_history.get(symbol, 0)
            # if signal_count > self.decay_limit:
            #     continue

            scored_assets.append(
                {
                    "symbol": symbol,
                    "score": risk_adj_mom,
                    "date": target_date,
                    "entry_price": current_price,
                }
            )

        if not scored_assets:
            return []

        # Sort and apply softmax weighting
        top_assets = sorted(scored_assets, key=lambda x: x["score"], reverse=True)[
            : self.top_n
        ]
        raw_scores = np.array([a["score"] for a in top_assets])
        weights = np.exp(raw_scores) / np.exp(raw_scores).sum()

        for i, asset in enumerate(top_assets):
            trade_intents.append(
                TradeIntent(
                    date=asset["date"],
                    legs=[
                        TradeIntentLeg(
                            symbol=asset["symbol"],
                            date=asset["date"],
                            signal="buy",
                            weight=weights[i],
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
        price_series = job.market_snapshot.get(
            symbol=symbol,
            variable="close",
            max_date=current_date,
            with_timestamps=True,
        )

        prices = pd.Series(OrderedDict(sorted(price_series.items())))
        prices.index = pd.to_datetime(prices.index)

        if current_date not in prices.index:
            return False

        current_price = prices.loc[current_date]
        ma = prices.rolling(window=self.ma_window).mean().loc[current_date]

        # Exit 1: Max holding period
        holding_days = (current_date - trade.legs[0].open_date).days
        if holding_days > self.max_holding_days:
            return True

        # Exit 2: Momentum reversal
        lookback = min(20, len(prices) - 1)
        if lookback >= 2:
            momentum = current_price / prices.iloc[-lookback] - 1
            if momentum < 0:
                return True

        # Exit 3: Trend break
        if current_price < ma:
            return True

        # Exit 4: ATR-based trailing stop-loss
        atr = self.compute_atr(prices, self.min_atr_window)
        entry_price = trade.legs[0].entry_price
        if current_price < entry_price - 1.5 * atr:
            return True

        return False
