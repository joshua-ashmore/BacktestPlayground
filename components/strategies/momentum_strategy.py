"""Momentum Strategy."""

from datetime import date
from typing import Dict, List, Literal, OrderedDict

import numpy as np
import pandas as pd

from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent, TradeIntentLeg
from components.trades.trade_model import Trade


class MomentumStrategyJob(Strategy):
    """Enhanced Momentum Strategy with risk-adjusted scoring, trend filtering, and softmax weights."""

    strategy_name: Literal["momentum"] = "momentum"
    base_window: int = 20
    top_n: int = 3
    momentum_threshold: float = 0.02
    ma_window: int = 50
    max_holding_days: int = 60
    min_atr_window: int = 14
    # decay_limit: int = 3  # Max number of recent signals allowed for same symbol

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
