"""Momentum Strategy."""

from datetime import date
from typing import Dict, List, Literal, OrderedDict

import numpy as np
import pandas as pd
from backtester.market_data.market import MarketSnapshot
from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent


class MomentumStrategyJob(Strategy):
    """Momentum Strategy Job."""

    strategy_name: Literal["momentum"] = "momentum"
    base_window: int = 20
    top_n: int = 3

    def generate_signals(
        self,
        job: StrategyJob,
        market_snapshot: MarketSnapshot,
        dates: List[date] | None = None,
    ) -> List[TradeIntent]:
        """Generate signals for all provided dates (or current_date if none given)."""
        dates = dates[1:] or [job.current_date]
        trade_intents: List[TradeIntent] = []

        for target_date in dates:
            momentum_scores = []

            for symbol in job.tickers:
                if symbol not in market_snapshot.data.keys():
                    continue
                price_series: Dict[str, float] = market_snapshot.get(
                    symbol=symbol,
                    variable="close",
                    with_timestamps=True,
                )

                # Ensure sorted by date
                sorted_prices = OrderedDict(sorted(price_series.items()))
                date_list = list(sorted_prices.keys())

                prices = pd.Series(sorted_prices)
                # returns = prices.pct_change()
                returns: pd.Series = np.log(prices).diff()
                # realized_vol = returns.rolling(20).std()
                # recent_vol = realized_vol.dropna().iloc[-1]
                # momentum_window = int(np.clip(self.base_window / recent_vol, 20, 90))

                # momentum_window = self.base_window

                # momentum_30 = prices.pct_change(30)
                # momentum_60 = prices.pct_change(60)
                # rank_30 = momentum_30.rank()
                # rank_60 = momentum_60.rank()
                # combined_rank = (rank_30 + rank_60) / 2
                # momentum_window = int(
                #     np.clip(self.base_window / combined_rank.autocorr(lag=1), 20, 90)
                # )

                realized_vol = returns.rolling(self.base_window).std()
                vol_percentile = (realized_vol.rank(pct=True)).iloc[-1]
                momentum_window = int(np.clip(90 - 70 * vol_percentile, 20, 90))

                # vol = returns.rolling(20).std().iloc[-1]
                # if vol < 0.015:
                #     window = 30
                # elif vol < 0.03:
                #     window = 50
                # elif vol < 0.045:
                #     window = 60
                # else:
                #     window = 80
                # momentum_window = window

                # # Step 1: Compute 20-day realized volatility
                # realized_vol = returns.rolling(self.base_window).std()

                # # Step 2: Calculate percentile of most recent vol
                # vol_percentile = realized_vol.rank(pct=True).iloc[-1]

                # # Step 3: Regime-based switching
                # if vol_percentile < 0.25:
                #     momentum_window = 30  # Low volatility → faster signal
                # elif vol_percentile < 0.50:
                #     momentum_window = 50
                # elif vol_percentile < 0.75:
                #     momentum_window = 60
                # else:
                #     momentum_window = 80  # High volatility → slower signal

                # returns = np.log(pd.Series(sorted_prices)).diff()
                # realized_vol = returns.rolling(20).std().mean()  # smoother estimate

                # # Invert vol to get longer window in calm markets, shorter in volatile
                # momentum_window = int(
                #     np.clip(self.base_window / (realized_vol + 1e-5), 20, 90)
                # )

                # returns = np.log(pd.Series(sorted_prices)).diff()
                # realized_vol_series = returns.rolling(20).std()
                # vol_percentile = realized_vol_series.rank(pct=True).iloc[-1]

                # # Use a steeper curve to bias toward default but still be adaptive
                # momentum_window = int(np.clip(90 - 70 * vol_percentile, 20, 90))

                # window_range = range(20, 90, 5)
                # best_r2 = 0
                # best_window = 30

                # log_prices = np.log(pd.Series(sorted_prices))

                # for w in window_range:
                #     if len(log_prices) < w:
                #         continue
                #     y = log_prices[-w:]
                #     x = np.arange(len(y))
                #     slope, intercept = np.polyfit(x, y, 1)
                #     y_fit = slope * x + intercept
                #     r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y.mean()) ** 2)

                #     if r2 > best_r2:
                #         best_r2 = r2
                #         best_window = w

                # momentum_window = best_window

                # returns = np.log(pd.Series(sorted_prices)).diff()
                # realized_vol = returns.rolling(20).std().mean()

                # candidate_windows = range(20, 90, 5)
                # best_score = -np.inf
                # best_window = 30

                # log_prices = np.log(pd.Series(sorted_prices))

                # for w in candidate_windows:
                #     if len(log_prices) < w:
                #         continue
                #     y = log_prices[-w:]
                #     x = np.arange(len(y))
                #     slope, intercept = np.polyfit(x, y, 1)
                #     y_fit = slope * x + intercept
                #     r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y.mean()) ** 2)

                #     # Reward high R² and penalize high volatility
                #     score = r2 / (realized_vol + 1e-4)

                #     if score > best_score:
                #         best_score = score
                #         best_window = w

                # momentum_window = best_window

                # returns = np.log(pd.Series(sorted_prices)).diff()
                # realized_vol = returns.rolling(20).std().mean()
                # momentum_window = int(np.clip(60 / (realized_vol + 1e-5), 20, 90))

                if target_date not in date_list:
                    continue

                idx = date_list.index(target_date)
                if idx < momentum_window:
                    continue

                try:
                    past_date = date_list[idx - momentum_window]
                    past_price = sorted_prices[past_date]
                    current_price = self.get_previous_price(
                        sorted_prices=sorted_prices, target_date=target_date
                    )

                    if past_price is None or current_price is None:
                        continue

                    momentum = current_price / past_price - 1
                    momentum_scores.append(
                        {"symbol": symbol, "momentum": momentum, "date": target_date}
                    )

                except (IndexError, KeyError):
                    continue

            # Rank and assign weights
            if not momentum_scores:
                continue

            ranked = sorted(momentum_scores, key=lambda x: x["momentum"], reverse=True)
            n = len(ranked)
            top = min(self.top_n, n)

            for i, symbol in enumerate(ranked):
                momentum_value = symbol["momentum"]

                if i < top and momentum_value > 0:
                    signal = "buy"
                    weight = 1.0 / top
                elif i >= n - top and momentum_value < 0:
                    signal = "sell"
                    weight = 1.0 / top
                else:
                    signal = "hold"
                    weight = 0.0

                trade_intents.append(
                    TradeIntent(
                        symbol=symbol["symbol"],
                        date=symbol["date"],
                        signal=signal,
                        weight=weight,
                        strategy=self.strategy_name,
                    )
                )

        return trade_intents
