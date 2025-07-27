"""Init."""

import datetime
from typing import List, Dict

from backtester.static_data import Directions
from backtester.strategy.strategies.statistical_arbitrage import (
    StatisticalArbitrageParameter,
    StatisticalArbitrageStrategyIntent,
)


class Trade:
    def __init__(
        self,
        entry_date,
        entry_signal: Directions,
        ticker_one_price,
        ticker_two_price,
        parameter: StatisticalArbitrageParameter,
        notional: float = 100,
    ):
        self.entry_date = entry_date
        self.signal = entry_signal
        self.parameter = parameter
        self.entry_price_one = ticker_one_price
        self.entry_price_two = ticker_two_price
        self.exit_date = None
        self.exit_price_one = None
        self.exit_price_two = None
        self.pnl = None
        self.notional = notional

    def close(
        self,
        exit_date: datetime.datetime,
        ticker_one_price: float,
        ticker_two_price: float,
    ):
        """Close trade."""
        self.exit_date = exit_date
        self.exit_price_one = ticker_one_price
        self.exit_price_two = ticker_two_price
        shares_one = self.notional / self.entry_price_one
        shares_two = shares_one * self.parameter.beta

        match self.signal:
            case Directions.BUY:
                # pnl_per_unit = (
                #     self.exit_price_one - self.entry_price_one
                # ) - self.parameter.beta * (self.exit_price_two - self.entry_price_two)
                pnl = shares_one * (
                    self.exit_price_one - self.entry_price_one
                ) - shares_two * (self.exit_price_two - self.entry_price_two)

            case Directions.SELL:
                # pnl_per_unit = -(
                #     self.exit_price_one - self.entry_price_one
                # ) + self.parameter.beta * (self.exit_price_two - self.entry_price_two)
                pnl = -shares_one * (
                    self.exit_price_one - self.entry_price_one
                ) + shares_two * (self.exit_price_two - self.entry_price_two)

            case _:
                # pnl_per_unit = 0.0
                pnl = 0.0

        # self.pnl = pnl_per_unit * self.notional
        self.pnl = pnl

    def is_open(self):
        return self.exit_date is None

    def calculate_pnl(self, ticker_one_price: float, ticker_two_price: float) -> float:
        # Replicate same logic as in .close() but don't update internal state
        direction = 1 if self.signal == Directions.BUY else -1
        return (
            direction
            * self.notional
            * (
                (ticker_one_price - self.entry_price_one)
                - (ticker_two_price - self.entry_price_two)
            )
        )

    def mark_to_market(self, ticker_one_price: float, ticker_two_price: float) -> float:
        """Estimate unrealized PnL as if closing now."""
        if self.signal == Directions.BUY:
            pnl = (ticker_two_price - self.entry_price_two) + (
                self.entry_price_one - ticker_one_price
            )
        elif self.signal == Directions.SELL:
            pnl = (self.entry_price_two - ticker_two_price) + (
                ticker_one_price - self.entry_price_one
            )
        else:
            pnl = 0.0
        return pnl * self.notional


# def execute_trades(intents: List[StatisticalArbitrageStrategyIntent]) -> Dict:
#     """Execute trades, tracking one open trade per ticker pair, closing only on CLOSE signals."""
#     trades: List[Trade] = []
#     open_trades: Dict[str, list[Trade]] = {}
#     daily_pnl = []

#     for intent in intents:
#         date = intent.entry_date
#         signal = intent.signal
#         ticker_one_price = float(intent.ticker_one_price)
#         ticker_two_price = float(intent.ticker_two_price)

#         # Create a unique key per ticker pair to track trades separately
#         pair_key = f"{intent.parameter.symbol_one}-{intent.parameter.symbol_two}"

#         # Skip invalid or no signal
#         if signal not in [Directions.BUY, Directions.SELL, Directions.CLOSE]:
#             daily_pnl.append(0.0)
#             continue

#         if signal in [Directions.BUY, Directions.SELL]:
#             if pair_key not in open_trades:
#                 open_trades[pair_key] = []
#             open_trades[pair_key].append(
#                 Trade(
#                     entry_date=date,
#                     entry_signal=signal,
#                     ticker_one_price=ticker_one_price,
#                     ticker_two_price=ticker_two_price,
#                     parameter=intent.parameter,
#                     notional=100,
#                 )
#             )
#             daily_pnl.append(0.0)  # No PnL on opening day

#         elif signal == Directions.CLOSE:
#             if pair_key in open_trades and open_trades[pair_key]:
#                 for trade in open_trades[pair_key][:]:
#                     trade.close(
#                         exit_date=date,
#                         ticker_one_price=ticker_one_price,
#                         ticker_two_price=ticker_two_price,
#                     )
#                     trades.append(trade)  # <-- Append here
#                     open_trades[pair_key].remove(trade)
#                     daily_pnl.append(trade.pnl if trade.pnl is not None else 0.0)
#                 if not open_trades[pair_key]:
#                     del open_trades[pair_key]
#             else:
#                 daily_pnl.append(0.0)

#     # Optional: Close any remaining open trades on last intent date (if you want)
#     last_date = intents[-1].entry_date if intents else None
#     if last_date is not None:
#         for pair_key, trades in open_trades.items():
#             for trade in trades:
#                 trade.close(
#                     exit_date=last_date,
#                     ticker_one_price=float(intents[-1].ticker_one_price),
#                     ticker_two_price=float(intents[-1].ticker_two_price),
#                 )
#                 # Add their PnL to the last daily pnl
#                 if daily_pnl:
#                     daily_pnl[-1] += trade.pnl if trade.pnl is not None else 0.0

#     # Compute stats
#     total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
#     num_trades = len(trades)
#     winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
#     losing_trades = sum(1 for t in trades if t.pnl is not None and t.pnl <= 0)
#     average_trade_pnl = (
#         sum(t.pnl for t in trades if t.pnl is not None) / num_trades
#         if num_trades > 0
#         else 0.0
#     )

#     return {
#         "trades": trades,
#         "daily_pnl": daily_pnl,
#         "total_pnl": total_pnl,
#         "num_trades": num_trades,
#         "winning_trades": winning_trades,
#         "losing_trades": losing_trades,
#         "average_trade_pnl": average_trade_pnl,
#     }


from datetime import datetime, timedelta


def execute_trades(
    intents: List[StatisticalArbitrageStrategyIntent], max_holding_days: int = 10
) -> Dict:
    trades: List[Trade] = []
    open_trades: Dict[str, list[Trade]] = {}
    daily_pnl = []

    for intent in intents:
        date = intent.entry_date
        signal = intent.signal
        ticker_one_price = float(intent.ticker_one_price)
        ticker_two_price = float(intent.ticker_two_price)

        pair_key = f"{intent.parameter.symbol_one}-{intent.parameter.symbol_two}"

        # 🔁 Enforce max holding period for all open trades
        # for pair, trade_list in list(open_trades.items()):
        #     for trade in trade_list[:]:  # Make a copy for safe removal
        #         days_open = (date - trade.entry_date).days
        #         if days_open > 5:
        #             current_pnl = trade.calculate_pnl(
        #                 ticker_one_price=ticker_one_price,
        #                 ticker_two_price=ticker_two_price,
        #             )
        #             if current_pnl > 0:
        #                 trade.close(
        #                     exit_date=date,
        #                     ticker_one_price=ticker_one_price,
        #                     ticker_two_price=ticker_two_price,
        #                 )
        #                 trades.append(trade)
        #                 trade_list.remove(trade)
        #                 daily_pnl.append(trade.pnl if trade.pnl is not None else 0.0)

        #     if not trade_list:
        #         del open_trades[pair]

        if signal in [Directions.BUY, Directions.SELL]:
            if pair_key not in open_trades:
                open_trades[pair_key] = []
            open_trades[pair_key].append(
                Trade(
                    entry_date=date,
                    entry_signal=signal,
                    ticker_one_price=ticker_one_price,
                    ticker_two_price=ticker_two_price,
                    parameter=intent.parameter,
                    notional=100,
                )
            )
            daily_pnl.append(0.0)

        elif signal == Directions.CLOSE:
            if pair_key in open_trades and open_trades[pair_key]:
                for trade in open_trades[pair_key][:]:
                    trade.close(
                        exit_date=date,
                        ticker_one_price=ticker_one_price,
                        ticker_two_price=ticker_two_price,
                    )
                    trades.append(trade)
                    open_trades[pair_key].remove(trade)
                    daily_pnl.append(trade.pnl if trade.pnl is not None else 0.0)
                if not open_trades[pair_key]:
                    del open_trades[pair_key]
            else:
                daily_pnl.append(0.0)

    # Final day close (optional but more realistic with max holding period)
    last_date = intents[-1].entry_date if intents else None
    if last_date is not None:
        for pair_key, trade_list in open_trades.items():
            for trade in trade_list:
                trade.close(
                    exit_date=last_date,
                    ticker_one_price=float(intents[-1].ticker_one_price),
                    ticker_two_price=float(intents[-1].ticker_two_price),
                )
                if daily_pnl:
                    daily_pnl[-1] += trade.pnl if trade.pnl is not None else 0.0
                trades.append(trade)

    # Compute stats
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
    num_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl is not None and t.pnl <= 0)
    average_trade_pnl = total_pnl / num_trades if num_trades > 0 else 0.0

    return {
        "trades": trades,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "average_trade_pnl": average_trade_pnl,
    }
