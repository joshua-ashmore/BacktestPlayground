"""Simple Execution Engine."""

from collections import defaultdict
from datetime import date
from typing import List, Literal, Tuple

from pydantic import model_validator

from components.execution.base_model import ExecutionEngineBase
from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade, TradeLeg


class SimpleExecutionEngine(ExecutionEngineBase):
    """Simple Execution Engine Model."""

    engine_type: Literal["simple"] = "simple"

    cash: float = 0
    initial_cash: float = 100_000

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.cash = self.initial_cash
        if self.positions is None:
            self.positions = defaultdict(float)
        return self

    def _open_trade(
        self,
        job: StrategyJob,
        intent: TradeIntent,
        symbol: Tuple[str, ...],
    ) -> List[Trade]:
        """Open new trade."""
        trade_legs = []
        for intent_leg in intent.legs:
            price_data = job.market_snapshot.get(
                symbol=intent_leg.symbol,
                variable="open",
                dates=intent.date,
                with_timestamps=True,
            )

            price = price_data.get(intent.date)
            if price is None:
                return []

            notional = intent_leg.weight * self.cash * self.allocation_pct_per_trade
            quantity = notional / price
            trade_legs.append(
                TradeLeg(
                    symbol=intent_leg.symbol,
                    open_date=intent_leg.date,
                    quantity=quantity,
                    price=price,
                    notional=notional,
                    strategy=intent_leg.strategy,
                    side=intent_leg.signal,
                )
            )
        if not len(trade_legs):
            return []
        trade = Trade(legs=trade_legs, metadata=intent.metadata)
        self.open_positions[symbol] = trade
        self.cash -= sum([leg.notional for leg in trade.legs])
        return [trade]

    def _close_trade(
        self,
        job: StrategyJob,
        target_date: date,
        strategy: Strategy,
        intent: TradeIntent,
        existing_trade: Trade,
        symbol: Tuple[str, ...],
    ):
        """Close existing trade."""
        close_signal = strategy.generate_exit_signals(
            job=job, trade=existing_trade, current_date=target_date
        )
        trade_age = (intent.date - existing_trade.legs[0].open_date).days

        if trade_age >= self.max_hold_days or close_signal:
            for leg in existing_trade.legs:
                price_data = job.market_snapshot.get(
                    symbol=leg.symbol,
                    variable="close",
                    dates=intent.date,
                    with_timestamps=True,
                )

                price = price_data.get(intent.date)
                if price is None:
                    return
                leg.close_date = intent.date
                leg.close_price = price
                leg.pnl = (price - leg.price) * leg.quantity * leg.direction_multiplier
                self.cash += leg.pnl + leg.notional
            del self.open_positions[symbol]

    def _stop_loss_close_check(self, job: StrategyJob, target_date: date):
        """Check for stop loss closing requirements."""
        for symbol, open_trade in list(self.open_positions.items()):
            pct_change = 0
            for leg in open_trade.legs:
                price_data = job.market_snapshot.get(
                    symbol=leg.symbol,
                    variable="close",
                    dates=target_date,
                    with_timestamps=True,
                )
                price = price_data.get(target_date)
                if price is None:
                    continue
                trade_age = (target_date - leg.open_date).days

                if leg.side == "buy":
                    pct_change += (price / leg.price) - 1
                else:
                    pct_change += (leg.price / price) - 1

            stop_hit = False
            if pct_change <= self.stop_loss_pct or pct_change >= self.take_profit_pct:
                stop_hit = True

            if stop_hit or trade_age >= self.max_hold_days:
                for leg in open_trade.legs:
                    price_data = job.market_snapshot.get(
                        symbol=leg.symbol,
                        variable="close",
                        dates=target_date,
                        with_timestamps=True,
                    )
                    price = price_data.get(target_date)
                    leg.close_date = target_date
                    leg.close_price = price
                    leg.pnl = (
                        (price - leg.price) * leg.quantity * leg.direction_multiplier
                    )
                    self.cash += leg.pnl + leg.notional
                del self.open_positions[symbol]
