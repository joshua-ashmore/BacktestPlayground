"""Simple Backtester Model."""

from collections import defaultdict
from datetime import date
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from components.backtester.base_model import BacktesterBaseModel, PortfolioSnapshot
from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade, TradeLeg


class Signal(str, Enum):
    """Signal Enum."""

    BUY = "buy"
    SELL = "sell"

    def is_opposite(self, other: "Signal") -> bool:
        return (self == Signal.BUY and other == Signal.SELL) or (
            self == Signal.SELL and other == Signal.BUY
        )


class SimpleBacktester(BacktesterBaseModel):
    """Simple Backtester Model."""

    backtester_type: Literal["simple"] = "simple"

    initial_cash: float = 100_000
    cash: float = 0
    portfolio_value_history: Dict = Field(default_factory=dict)
    position_book: Dict = Field(default_factory=dict)

    max_hold_days: int = 5
    allocation_pct_per_trade: float = 0.1
    stop_loss_pct: float = -0.03
    take_profit_pct: float = 0.06

    trades: List[Trade] = Field(default_factory=list)
    open_positions: Dict[Tuple[str, ...], Trade] = Field(default_factory=dict)

    equity_curve: List[Trade] = Field(default_factory=list)
    positions: Optional[Dict] = Field(default=None)

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.cash = self.initial_cash
        if self.positions is None:
            self.positions = defaultdict(float)
        return self

    def simulate_trade_execution(
        self,
        job: StrategyJob,
        dates: List[date],
    ) -> list[Trade]:
        """Simulate trade execution across dates."""
        for target_date in dates:
            self.trades.append(
                self.simulate_trade_execution_on_date(
                    job=job,
                    target_date=target_date,
                    previous_date=dates[-2],
                )
            )
        return self.trades

    def simulate_trade_execution_on_date(
        self,
        job: StrategyJob,
        target_date: date,
        previous_date: date,
        strategy: Strategy,
    ):
        """Simulate trade execution on date."""
        trades: List[Trade] = []
        intents_on_date = [i for i in job.signals if i.date == target_date]
        for intent in intents_on_date:
            symbol = tuple([intent_leg.symbol for intent_leg in intent.legs])
            existing_trade = self.open_positions.get(symbol)

            if not existing_trade:
                trades.extend(self._open_trade(job=job, intent=intent, symbol=symbol))
            else:
                self._close_trade(
                    job=job,
                    target_date=target_date,
                    strategy=strategy,
                    intent=intent,
                    existing_trade=existing_trade,
                    symbol=symbol,
                )

        self._stop_loss_close_check(job=job, target_date=target_date)
        return trades

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
        trade_age = (intent.date - existing_trade.legs[0].open_date).days

        # trade_pnl = 0
        # for leg in existing_trade.legs:
        #     price_data = job.market_snapshot.get(
        #         symbol=leg.symbol,
        #         variable="close",
        #         dates=intent.date,
        #         with_timestamps=True,
        #     )

        #     price = price_data.get(intent.date)
        #     if price is None:
        #         return

        #     if leg.side == "buy":
        #         trade_pnl += (price - leg.price) * leg.quantity
        #     else:
        #         trade_pnl += (leg.price - price) * leg.quantity

        close_signal = strategy.generate_exit_signals(
            job=job, trade=existing_trade, current_date=target_date
        )

        if (
            # Signal(intent.signal).is_opposite(Signal(existing_trade.side))
            trade_age >= self.max_hold_days
            or close_signal
        ):
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

                # self.cash += price * leg.quantity
                leg.pnl = (price - leg.price) * leg.quantity * leg.direction_multiplier
                self.cash += leg.pnl + leg.notional
            del self.open_positions[symbol]

    def run(
        self,
        job: StrategyJob,
        dates: list[date],
    ) -> List[PortfolioSnapshot]:
        """
        Rebuild the equity curve from trades and market data.
        Assumes trades are sorted by date.
        """
        for target_date in dates:
            self.equity_curve.append(self.run_on_date(job=job, target_date=target_date))
        return self.equity_curve

    def run_on_date(
        self,
        job: StrategyJob,
        target_date: date,
    ) -> PortfolioSnapshot:
        """Run on date."""
        trades = job.equity_curve
        cash = self.cash
        trade_mtm = 0
        trade_notional = 0

        # Positions of trades closed today
        for trade in trades:
            for leg in trade.legs:
                if leg.close_date == target_date:
                    self.positions[leg.symbol] -= leg.quantity

        # Positions of trades opened today
        trades_today = [t for t in trades if t.legs[0].open_date == target_date]
        for trade in trades_today:
            for leg in trade.legs:
                self.positions[leg.symbol] += leg.quantity

        # Daily Mark-to-Market value
        prices = {
            sym: job.market_snapshot.get(
                symbol=sym, variable="close", dates=target_date, with_timestamps=True
            ).get(target_date)
            for sym in self.positions.keys()
        }
        for trade in job.equity_curve:
            for leg in trade.legs:
                if leg.close_date is None:
                    # if leg.open_date <= target_date and (
                    #     leg.close_date is None or leg.close_date >= target_date
                    # ):
                    trade_mtm += (
                        (prices.get(leg.symbol) - leg.price)
                        * leg.direction_multiplier
                        * leg.quantity
                    )
                    trade_notional += leg.notional

        snapshot = PortfolioSnapshot(
            date=target_date,
            total_value=cash + trade_mtm + trade_notional,
            cash=cash,
            positions=dict(self.positions),
            prices={k: v for k, v in prices.items() if v is not None},
        )
        return snapshot
