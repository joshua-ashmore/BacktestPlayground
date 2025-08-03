"""Naive Regime Detection Engine."""

from datetime import date
from typing import Dict, Literal

import numpy as np
import pandas as pd
from pydantic import model_validator

from components.job.base_model import StrategyJob
from components.regime.base_model import AbstractRegimeEngine
from components.strategies.fallback_strategy import FallbackStrategyJob


class NaiveRegimeEngine(AbstractRegimeEngine):
    """Regime Detection Engine."""

    engine_type: Literal["naive"] = "naive"

    window_short: int = 10
    window_long: int = 60
    adx_window: int = 14
    group_map: dict = {}

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.strategy_map["fallback"] = FallbackStrategyJob(symbols=[])
        return self

    def calculate_market_breadth(self, price_data: pd.Series):
        # % of assets above their 50-day moving average (breadth indicator)
        ma50 = price_data.rolling(50).mean()
        breadth = (price_data > ma50).sum(axis=1) / price_data.shape[1]
        return breadth

    def detect_regime(self, job: StrategyJob, target_date: date, previous_date: date):
        """Detect regime."""
        regime_probs = self.detect_regime_probabilities(
            job=job, previous_date=previous_date
        )
        top_regime = max(regime_probs, key=regime_probs.get)
        self.regime_history[target_date] = top_regime
        strategy = self.strategy_map[top_regime]
        print(f"{target_date}: {top_regime}")
        job.tickers = strategy.symbols
        return strategy

    def detect_regime_old(
        self, job: StrategyJob, target_date: date, previous_date: date
    ):
        """
        price_data: DataFrame of daily close prices, columns=assets
        Returns a regime label string
        """
        # Calculate market index (equal weighted mean of prices)
        price_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="close"
        )

        # Calculate market breadth
        breadth = self.calculate_market_breadth(price_data).iloc[-1]
        market = price_data.mean(axis=1)

        # Calculate returns
        returns = market.pct_change()

        # Multi-window rolling mean returns
        rolling_return_long = returns.rolling(self.window_long).mean().iloc[-1]

        # Multi-window rolling volatility (std)
        rolling_vol_long = returns.rolling(self.window_long).std().iloc[-1]

        # Calculate RSI on market price to identify momentum (period=14)
        rsi = self.calculate_rsi(series=market, period=self.adx_window).iloc[-1]

        # Calculate ADX to measure trend strength (period=adx_window)
        high_price_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="high"
        )
        low_price_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="low"
        )
        adx = self.calculate_adx(
            high=high_price_data.mean(axis=1),
            low=low_price_data.mean(axis=1),
            close=market,
            period=self.adx_window,
        ).iloc[-1]

        # Adaptive volatility thresholds based on historical quantiles (last 250 days)
        vol_hist = returns.rolling(self.window_long).std().dropna()
        vol_low_thresh = vol_hist.quantile(0.25)
        vol_high_thresh = vol_hist.quantile(0.75)

        # Regime decision logic
        if (
            rolling_return_long > 0
            and adx > 25
            and rsi > 50
            and rolling_vol_long < vol_high_thresh
        ):
            regime = "trending"
        elif rolling_vol_long > vol_high_thresh:
            regime = "volatile"
        elif (
            abs(rolling_return_long) < 0.001
            and rolling_vol_long < vol_low_thresh
            and rsi < 50
        ):
            regime = "mean_reverting"
        elif breadth > 0.6 and rolling_return_long > 0:
            # Market breadth confirms healthy uptrend, fallback to trending
            regime = "trending"
        else:
            regime = "fallback"
        self.regime_history[target_date] = regime
        strategy = self.strategy_map[regime]
        print(f"{target_date}: {regime}")
        job.tickers = strategy.symbols
        return strategy

    def calculate_rsi(self, series: pd.Series, period: int = 14):
        """Calulate RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use Wilder's smoothing method after initial avg gain/loss
        avg_gain = avg_gain.combine_first(
            gain.ewm(alpha=1 / period, adjust=False).mean()
        )
        avg_loss = avg_loss.combine_first(
            loss.ewm(alpha=1 / period, adjust=False).mean()
        )

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ):
        """Calculate ADX."""
        # True Range (TR)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smooth TR, +DM, -DM with Wilder's smoothing
        tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        dx = dx.replace([np.inf, -np.inf], 0).fillna(0)

        # ADX is smoothed DX
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx

    def get_aligned_prices(
        self, job: StrategyJob, previous_date: date, variable: str
    ) -> pd.DataFrame:
        price_series = {}
        for symbol in job.market_snapshot.data.keys():
            closes = job.market_snapshot.get(
                symbol=symbol,
                variable=variable,
                max_date=previous_date,
                with_timestamps=True,
            )
            series = pd.Series(
                data=list(closes.values()), index=pd.to_datetime(list(closes.keys()))
            ).sort_index()
            price_series[symbol] = series

        common_index = sorted(
            set.intersection(*(set(s.index) for s in price_series.values()))
        )
        aligned_data = {
            symbol: s.reindex(common_index) for symbol, s in price_series.items()
        }
        prices_df = pd.DataFrame(aligned_data)

        return prices_df

    def calculate_group_index(self, price_data: pd.Series, group: dict):
        cols = [
            col
            for col in price_data.columns
            if self.group_map.get(col, "default") == group
        ]
        return price_data[cols].mean(axis=1)

    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def detect_regime_probabilities(
        self, job: StrategyJob, previous_date: date
    ) -> Dict:
        # Handle multiple groups (e.g., 'crypto', 'equity')
        groups = set(self.group_map.values()) if self.group_map else {"default"}
        regimes = ["trending", "volatile", "mean_reverting", "fallback"]
        scores = {r: 0.0 for r in regimes}

        price_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="close"
        )
        high_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="high"
        )
        low_data = self.get_aligned_prices(
            job=job, previous_date=previous_date, variable="low"
        )

        for group in groups:
            market = self.calculate_group_index(price_data, group)
            high = self.calculate_group_index(high_data, group)
            low = self.calculate_group_index(low_data, group)
            returns = market.pct_change()
            rolling_return_long = returns.rolling(self.window_long).mean().iloc[-1]
            rolling_vol_long = returns.rolling(self.window_long).std().iloc[-1]

            # Simulated high/low for ADX (approximation)

            close = market

            adx_val = self.calculate_adx(high, low, close, period=self.adx_window).iloc[
                -1
            ]
            rsi_val = self.calculate_rsi(market, period=14).iloc[-1]

            # Normalize indicators for scoring (z-score style)
            score_trending = (
                max(0, rolling_return_long * 100)
                + max(0, (rsi_val - 50))
                + max(0, adx_val - 25)
                - max(0, rolling_vol_long * 100)
            )
            score_volatile = max(0, rolling_vol_long * 100 - 1.5)
            score_mean_reverting = (
                max(0, (50 - rsi_val))
                + max(0, 1.5 - rolling_vol_long * 100)
                - abs(rolling_return_long * 100)
            )
            score_fallback = 1.0  # Bias to ensure fallback has a baseline

            # Aggregate group-wise (sum)
            scores["trending"] += score_trending
            scores["volatile"] += score_volatile
            scores["mean_reverting"] += score_mean_reverting
            scores["fallback"] += score_fallback

        # Normalize using softmax to get probabilities
        raw_scores = np.array([scores[r] for r in regimes])
        probs = self.softmax(raw_scores)
        return dict(zip(regimes, probs))

    # def detect_regime(
    #     self, job: StrategyJob, target_date: date, previous_date: date
    # ) -> Strategies:
    #     """Detect regime."""
    #     prices_df = self.get_aligned_close_prices(job=job, previous_date=previous_date)
    #     regime = self.detect(price_data=prices_df)
    # self.regime_history[target_date] = regime
    # strategy = self.strategy_map[regime]
    # print(f"{target_date}: {regime}")
    # job.tickers = strategy.symbols
    # return strategy

    # def detect(self, price_data: pd.DataFrame) -> str:
    #     """
    #     Detects regime using trend and volatility logic.
    #     Assumes price_data is a DataFrame of daily close prices.
    #     """
    #     # Calculate average market (equal-weight index)
    #     market = price_data.mean(axis=1)

    #     # Rolling returns and rolling volatility
    #     returns = market.pct_change()
    #     rolling_return = returns.rolling(self.window).mean().iloc[-1]
    #     rolling_vol = returns.rolling(self.window).std().iloc[-1]

    #     # Thresholds (tunable)
    #     if rolling_return > 0.002 and rolling_vol < 0.015:
    #         return "trending"
    #     elif rolling_vol > 0.025:
    #         return "volatile"
    #     elif abs(rolling_return) < 0.0005 and rolling_vol < 0.012:
    #         return "mean_reverting"
    #     # elif market.pct_change(250).iloc[-1] > 0.10:
    #     #     return "inflation"
    #     else:
    #         # return "defensive"
    #         return "fallback"
