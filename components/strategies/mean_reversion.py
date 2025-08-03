"""Mean Reversion Strategy Model."""

from datetime import date
from itertools import combinations
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS

from components.trades.trade_model import Trade


class PairsTradingStrategyJob(Strategy):
    """Mean Reversion Strategy Model."""

    strategy_name: Literal["mean reversion"] = "mean reversion"
    window: int = 60
    z_score_threshold: float = 1.5
    min_hurst: float = 0.4
    min_coint_pval: float = 0.01
    min_adf_pval: float = 0.01

    def generate_signal_on_date(
        self, job: StrategyJob, target_date: date, previous_date: date
    ) -> List[TradeIntent]:
        trade_intents: List[TradeIntent] = []
        pair_dict: Dict = {}

        tickers = [
            ticker for ticker in job.tickers if ticker in job.market_snapshot.data
        ]
        pairs = list(combinations(tickers, 2))

        for sym1, sym2 in pairs:
            pair_signals: List[dict] = []
            series1 = job.market_snapshot.get(sym1, "close", max_date=previous_date)[
                -self.window :
            ]
            series2 = job.market_snapshot.get(sym2, "close", max_date=previous_date)[
                -self.window :
            ]

            if len(series1) < self.window or len(series2) < self.window:
                continue

            # Test both directions
            coint_results = []
            for a, b in [(series1, series2), (series2, series1)]:
                try:
                    _, p_value, _ = coint(a, b)
                    coint_results.append((p_value, a, b))
                except Exception:
                    continue

            if not coint_results:
                continue

            # Choose best cointegration direction
            best_pval, best_a, best_b = min(coint_results, key=lambda x: x[0])
            if best_pval >= self.min_coint_pval:
                continue

            # Regression to get beta (hedge ratio)
            beta = OLS(best_a, best_b).fit().params[0]

            spread = pd.Series(best_a - beta * np.array(best_b))

            # ADF test
            try:
                adf_stat, adf_pval, *_ = adfuller(spread)
            except Exception:
                continue
            if adf_pval >= self.min_adf_pval:
                continue

            # Hurst exponent filter
            hurst = self.hurst_exponent(spread)
            if hurst >= self.min_hurst:
                continue

            # Z-score
            # spread_mean = spread.mean()
            # spread_std = spread.std()
            # z_score = (spread[-1] - spread_mean) / spread_std
            spread_mean = spread.ewm(span=20).mean().iloc[-1]
            spread_std = spread.ewm(span=20).std().iloc[-1]
            z_score = (spread.iloc[-1] - spread_mean) / spread_std

            # Signal generation
            if z_score > self.z_score_threshold:
                pair_signals.append(
                    {
                        "symbol": sym1,
                        "signal": "sell",
                        "weight": abs(beta) / (1 + abs(beta)),
                    }
                )
                pair_signals.append(
                    {"symbol": sym2, "signal": "buy", "weight": 1 / (1 + abs(beta))}
                )
                pair_dict[(sym1, sym2)] = (
                    pair_signals,
                    {
                        "beta": beta,
                        "spread_mean": spread_mean,
                        "spread_std": spread_std,
                        "spread": spread,
                    },
                )
            elif z_score < -self.z_score_threshold:
                pair_signals.append(
                    {"symbol": sym1, "signal": "buy", "weight": 1 / (1 + abs(beta))}
                )
                pair_signals.append(
                    {
                        "symbol": sym2,
                        "signal": "sell",
                        "weight": abs(beta) / (1 + abs(beta)),
                    }
                )
                pair_dict[(sym1, sym2)] = (
                    pair_signals,
                    {
                        "beta": beta,
                        "spread_mean": spread_mean,
                        "spread_std": spread_std,
                        "spread": list(spread),
                    },
                )

        for _, (signals, metadata) in pair_dict.items():
            trade_intent_legs = [
                TradeIntentLeg(
                    symbol=signal["symbol"],
                    date=target_date,
                    signal=signal["signal"],
                    weight=signal["weight"],
                    strategy=self.strategy_name,
                )
                for signal in signals
            ]
            trade_intents.append(
                TradeIntent(date=target_date, legs=trade_intent_legs, metadata=metadata)
            )

        return trade_intents

    def hurst_exponent(self, time_series: Union[pd.Series, np.ndarray]) -> float:
        """Estimate the Hurst exponent to check for mean reversion behavior."""
        ts = np.asarray(time_series).flatten()

        lags = range(2, 20)
        tau = []
        log_lags = []

        for lag in lags:
            # Calculate lagged differences
            diff = ts[lag:] - ts[:-lag]
            std = np.std(diff)

            # Avoid divide-by-zero or log(0)
            if std > 1e-8:
                tau.append(std)
                log_lags.append(np.log(lag))

        if len(tau) < 2:
            raise ValueError("Not enough valid lags to compute Hurst exponent.")

        log_tau = np.log(tau)
        # Linear regression (slope = Hurst exponent)
        poly = np.polyfit(log_lags, log_tau, 1)
        hurst = poly[0]
        return hurst

    def generate_exit_signals(
        self, job: StrategyJob, trade: Trade, current_date: date
    ) -> bool:
        """Exit when spread z-score crosses zero or small threshold."""

        leg1, leg2 = trade.legs

        prices_1 = job.market_snapshot.get(
            leg1.symbol, variable="close", max_date=current_date
        )[-self.window :]
        prices_2 = job.market_snapshot.get(
            leg2.symbol, variable="close", max_date=current_date
        )[-self.window :]

        if len(prices_1) < self.window or len(prices_2) < self.window:
            return False

        # Use original beta from trade (stored at entry), not recalculated
        beta = trade.metadata.get("beta", 1.0)

        spread = pd.Series(np.array(prices_1) - beta * np.array(prices_2))
        # mean = np.mean(spread)
        # std = np.std(spread)

        spread_mean = spread.ewm(span=20).mean().iloc[-1]
        spread_std = spread.ewm(span=20).std().iloc[-1]
        z_score = (spread.iloc[-1] - spread_mean) / spread_std

        # if spread_std == 0:
        #     return False

        # z_score = (spread[-1] - mean) / spread_std

        # Exit if spread has reverted to near mean (or fully crossed it)
        return abs(z_score) < 0.5
