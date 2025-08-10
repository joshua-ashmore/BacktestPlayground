"""Execution Engine Base Model."""

from datetime import date
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from components import PortfolioSnapshot
from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade


class ExecutionEngineBase(BaseModel):
    """Execution Engine Base Model."""

    # TODO: add config (include slippage etc.)
    max_hold_days: int = 5
    allocation_pct_per_trade: float = 0.1
    stop_loss_pct: float = -0.03
    take_profit_pct: float = 0.06

    trades: List[Trade] = Field(default_factory=list)
    positions: Optional[Dict] = Field(default=None)
    open_positions: Dict[Tuple[str, ...], Trade] = Field(default_factory=dict)
    equity_curve: List[Trade] = Field(default_factory=list)

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
    ) -> List[Trade]:
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
        raise NotImplementedError("Not implemented method for stop_loss_check.")

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

    def _open_trade(
        self,
        job: StrategyJob,
        intent: TradeIntent,
        symbol: Tuple[str, ...],
    ) -> List[Trade]:
        """Open new trade."""
        raise NotImplementedError("Not yet implemented open_trade method.")

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
        raise NotImplementedError("Not yet implemented close_trade method.")
