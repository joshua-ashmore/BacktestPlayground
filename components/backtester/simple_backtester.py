"""Simple Backtester Model."""

from collections import defaultdict
from datetime import date
from enum import Enum
from typing import List, Literal

from pydantic import model_validator

from backtester.market_data.market import MarketSnapshot
from components.backtester.base_model import BacktesterBaseModel, PortfolioSnapshot
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade


class Signal(str, Enum):
    """Signal Enum."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    def is_opposite(self, other: "Signal") -> bool:
        return (self == Signal.BUY and other == Signal.SELL) or (
            self == Signal.SELL and other == Signal.BUY
        )


class SimpleBacktester(BacktesterBaseModel):
    """Simple Backtester Model."""

    backtester_type: Literal["simple"] = "simple"

    initial_cash: float = 100_000
    cash: float = 0
    portfolio_value_history: dict = {}
    position_book: dict = {}  # {symbol: units held}

    max_hold_days: int = 5
    allocation_pct_per_trade: float = 0.1
    stop_loss_pct: float = -0.03
    take_profit_pct: float = 0.06

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.cash = self.initial_cash
        return self

    def simulate_trade_execution(
        self,
        intents: list[TradeIntent],
        market_snapshot: MarketSnapshot,
        dates: list[date],
    ) -> list[Trade]:
        trades: list[Trade] = []
        open_positions: dict[str, Trade] = {}

        for _date in dates:
            # 1) Process intents on this date to open or close trades by signal
            intents_on_date = [i for i in intents if i.date == _date]
            for intent in intents_on_date:
                # (existing open/close logic here, but WITHOUT stop loss check yet)

                price_data = market_snapshot.get(
                    symbol=intent.symbol,
                    variable="close",
                    dates=intent.date,
                    with_timestamps=True,
                )

                price = price_data.get(intent.date)
                if price is None or intent.signal == Signal.HOLD:
                    continue

                symbol = intent.symbol
                existing_trade = open_positions.get(symbol)

                # CASE 1: No open trade — open one
                if not existing_trade:
                    if intent.signal != Signal.HOLD:
                        notional = (
                            intent.weight * self.cash * self.allocation_pct_per_trade
                        )
                        quantity = notional / price

                        trade = Trade(
                            symbol=symbol,
                            open_date=intent.date,
                            quantity=quantity,
                            price=price,
                            notional=quantity * price,
                            strategy=intent.strategy,
                            side=intent.signal,
                        )
                        trades.append(trade)
                        open_positions[symbol] = trade

                        self.cash -= notional

                # CASE 2: Trade is open — check if we should close
                else:
                    trade_age = (intent.date - existing_trade.open_date).days
                    side = existing_trade.side

                    if side == "buy":
                        trade_pnl = (
                            price - existing_trade.price
                        ) * existing_trade.quantity
                    else:
                        trade_pnl = (
                            existing_trade.price - price
                        ) * existing_trade.quantity

                    if (
                        Signal(intent.signal).is_opposite(Signal(existing_trade.side))
                        or trade_age >= self.max_hold_days
                    ):
                        existing_trade.close_date = intent.date
                        existing_trade.close_price = price
                        existing_trade.pnl = trade_pnl
                        del open_positions[symbol]

                        self.cash += price * existing_trade.quantity

            # 2) For all currently open trades, check stop loss and max hold days
            for symbol, open_trade in list(open_positions.items()):
                # Get current price for this symbol on this date
                price_data = market_snapshot.get(
                    symbol=symbol, variable="close", dates=_date, with_timestamps=True
                )
                price = price_data.get(_date)
                if price is None:
                    continue  # no price data, skip

                trade_age = (_date - open_trade.open_date).days

                side = open_trade.side
                if side == "buy":
                    trade_pnl = (price - open_trade.price) * open_trade.quantity
                    pct_change = (price / open_trade.price) - 1
                else:
                    trade_pnl = (open_trade.price - price) * open_trade.quantity
                    pct_change = (open_trade.price / price) - 1

                stop_hit = False
                if side == "buy" and (
                    pct_change <= self.stop_loss_pct
                    or pct_change >= self.take_profit_pct
                ):
                    stop_hit = True
                elif side == "sell" and (
                    pct_change <= self.stop_loss_pct
                    or pct_change >= self.take_profit_pct
                ):
                    stop_hit = True

                if stop_hit or trade_age >= self.max_hold_days:
                    open_trade.close_date = _date
                    open_trade.close_price = price
                    open_trade.pnl = trade_pnl
                    del open_positions[symbol]
                    self.cash += price * open_trade.quantity

        return trades

    def run(
        self,
        trades: List[Trade],
        market_snapshot: MarketSnapshot,
        dates: list[date],
    ) -> List[PortfolioSnapshot]:
        """
        Rebuild the equity curve from trades and market data.
        Assumes trades are sorted by date.
        """
        equity_curve = []
        cash = self.initial_cash
        positions = defaultdict(float)  # symbol -> quantity

        for dt in dates:
            # 2. Close trades closed today (optional if you're modeling closing)
            trades_closed_today = [t for t in trades if t.close_date == dt]
            for trade in trades_closed_today:
                # You could adjust cash and positions if needed
                cash += trade.quantity * market_snapshot.get(
                    symbol=trade.symbol,
                    variable="close",
                    dates=dt,
                    with_timestamps=True,
                ).get(dt)
                positions[trade.symbol] -= trade.quantity

            # 1. Execute trades opened today
            trades_today = [t for t in trades if t.open_date == dt]
            for trade in trades_today:
                cash -= trade.notional
                positions[trade.symbol] += trade.quantity

            # 3. Mark-to-market positions
            prices = {
                sym: market_snapshot.get(
                    symbol=sym, variable="close", dates=dt, with_timestamps=True
                ).get(dt)
                for sym in positions.keys()
            }

            total_value = cash + sum(
                qty * price
                for sym, qty in positions.items()
                if (price := prices.get(sym)) is not None
            )

            snapshot = PortfolioSnapshot(
                date=dt,
                total_value=total_value,
                cash=cash,
                positions=dict(positions),
                prices={k: v for k, v in prices.items() if v is not None},
            )
            equity_curve.append(snapshot)

        return equity_curve
