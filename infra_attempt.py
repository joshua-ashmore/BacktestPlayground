"""Infrastructure Re-write Attempt."""

from datetime import date, timedelta
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

from backtester.market_data.market import MarketSnapshot
from backtester.market_data.market_data_feed import MarketFeed


class Directions(str, Enum):
    """Directions Enum."""

    BUY = "Buy"
    SELL = "Sell"


class Trade(BaseModel):
    """Trade Model."""

    # Entry Attributes
    entry_date: date
    entry_price: float
    direction: Directions
    notional_currency: str
    contract: str
    notional_amount: Optional[float] = None
    number_of_contracts: Optional[int] = None

    # Additional Attributes
    transaction_cost: float = 0
    slippage: float = 0

    # Exit Attributes
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        if not (self.notional_amount and self.number_of_contracts):
            raise ValidationError(
                f"Need provide one of {self.notional_amount} or {self.number_of_contracts}."
            )

        if not self.notional_amount:
            self.calculate_notional_amount()
        if not self.number_of_contracts:
            self.calculate_number_of_contracts()
        return self

    def calculate_notional_amount(self):
        """Calculate notional amount."""
        self.notional_amount = self.number_of_contracts * self.entry_price

    def calculate_number_of_contracts(self):
        """Calculate number of contracts."""
        self.number_of_contracts = self.notional_amount / self.entry_price
        self.calculate_notional_amount()

    def direction_multiplier(self):
        """Calculate direction multiplier."""
        match self.direction:
            case Directions.BUY:
                return 1
            case Directions.SELL:
                return -1

    def calculate_payoff(self):
        """Calculate payoff."""
        return (
            self.exit_price - self.entry_price - self.slippage
        ) * self.number_of_contracts * self.direction_multiplier() - self.transaction_cost

    def calculate_mtm(self):
        """Calculate mark-to-market."""
        return (
            (self.exit_price - self.entry_price)
            * self.number_of_contracts
            * self.direction_multiplier()
        )


"""Trade Intents."""


class Signals(str, Enum):
    """Signals Enum."""

    BUY = "Buy"
    SELL = "Sell"
    NONE = "None"
    HOLD = "Hold"


class TradeIntent(BaseModel):
    """Trade Intent."""

    # Intent Attributes
    symbol: str
    intent_date: date
    signal: Signals

    # Capital Allocation
    notional_currency: str
    notional_amount: float = 0

    # Allocation
    confidence: Optional[float] = None
    signal_strength: Optional[float] = None


"""Strategy Logic."""


class AbstractStrategy(BaseModel):
    """Abstract Strategy Model."""

    strategy: Literal["Abstract"] = "Abstract"

    def evaluate(self, current_date: date, market_snapshot: MarketSnapshot) -> Signals:
        """Evaluates strategy on current date."""
        raise NotImplementedError(
            f"Not implemented evaluate function for {self.strategy}."
        )

    def create_intent(
        self,
        current_date: date,
        market_snapshot: MarketSnapshot,
        signal: Signals,
    ) -> TradeIntent:
        """Create intent."""
        raise NotImplementedError(
            f"Not implemented create intent function for {self.strategy}."
        )


PriceDataVariables = Union[Literal["close"], Literal["open"]]


class MovingAverageStrategyConfig(BaseModel):
    """Moving Average Strategy Config."""

    symbol: str
    price_data_variable: PriceDataVariables = Field(
        description="Price data variable to use for window.", examples=["close"]
    )
    short_window: int = Field(description="Short moving average window length.")
    long_window: int = Field(description="Long moving average window length.")

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        if self.short_window >= self.long_window:
            raise ValueError(
                f"Short window length: {self.short_window} must be shorted than long window length: {self.long_window}."
            )
        return self


class MovingAverageStrategy(AbstractStrategy):
    """Moving Average Strategy."""

    strategy: Literal["Moving Average"] = "Moving Average"
    config: MovingAverageStrategyConfig

    def evaluate(self, current_date: date, market_snapshot: MarketSnapshot) -> Signals:
        """Evaluates moving average strategy on current date."""
        price_data = pd.Series(
            market_snapshot.get(
                symbol=self.config.symbol,
                variable=self.config.price_data_variable,
                min_date=current_date - timedelta(days=self.config.long_window),
                max_date=current_date - timedelta(days=1),
                with_timestamps=False,
            )
        )

        if len(price_data) < self.config.long_window:
            return Signals.HOLD

        long_ma = price_data.rolling(window=self.config.long_window).mean()
        short_ma = price_data.rolling(window=self.config.short_window).mean()

        if short_ma.isna().any() or long_ma.isna().any():
            return Signals.HOLD

        previous_short = short_ma[-2]
        previous_long = long_ma[-2]
        current_short = short_ma[-1]
        current_long = long_ma[-1]

        if previous_short < previous_long and current_short > current_long:
            return Signals.BUY  # Bullish crossover
        elif previous_short > previous_long and current_short < current_long:
            return Signals.SELL  # Bearish crossover
        else:
            return Signals.HOLD

    def generate_intent(
        self,
        current_date: date,
        market_snapshot: MarketSnapshot,
        signal: Signals,
    ) -> TradeIntent:
        """Generate intent."""
        match signal:
            case Signals.BUY | Signals.SELL:
                return TradeIntent(
                    symbol=self.config.symbol, intent_date=current_date, signal=signal
                )
            case Signals.HOLD | Signals.NONE:
                return None


AllStrategies = Annotated[
    Union[AbstractStrategy, MovingAverageStrategy], Field(..., discriminator="strategy")
]


"""Backtesting Results."""


class NotionalAllocationRules(str, Enum):
    """Notional Allocation Rules Enum."""

    EQUAL = "equal"


class PortfolioState(BaseModel):
    """Portfolio State."""

    cash: float = 100_000.0
    trades: list[Trade] = []

    def filter_trade_intents(
        self,
        todays_intents: list[TradeIntent],
        notional_allocation_rule: NotionalAllocationRules,
    ) -> list[TradeIntent]:
        """Filter trade intents."""
        # Do we need some rule here?
        match notional_allocation_rule:
            case NotionalAllocationRules.EQUAL:
                for intent in todays_intents:
                    intent.notional_amount = self.cash / len(todays_intents)
        return todays_intents

    def execute_trade(
        self, intent: TradeIntent, current_date: date, market_snapshot: MarketSnapshot
    ) -> Trade:
        """Execute trade."""
        notional_amount = 100
        return Trade(
            entry_date=current_date,
            entry_price=market_snapshot.get(
                symbol=intent.symbol,
                variable="close",
                dates=current_date,
                with_timestamps=False,
            ),
            direction=intent.signal,
            contract=intent.symbol,
            notional_amount=notional_amount,
            notional_currency="USD",
        )

    def evaluate_closures(self, current_date: date, market_snapshot: MarketSnapshot):
        """Evaluate close positions."""
        # Close trades and add notional back to portfolio.
        for trade in self.trades:
            # if should exit
            trade.exit_date = current_date
            trade.exit_price = market_snapshot.get(
                symbol=trade.contract,
                variable="close",
                dates=current_date,
                with_timestamps=False,
            )
            self.cash += trade.calculate_payoff()

    def compute_metrics(self):
        """Compute portfolio metrics."""
        return 0


class BacktestResult(BaseModel):
    """Backtest Results Model."""

    executed_trades: list[Trade]
    portfolio: list
    performance: Any


"""Backtesting Engine."""


class Engine(BaseModel):
    """Engine Model."""

    start_date: date

    def generate_trading_intents(
        self,
        market_snapshot: MarketSnapshot,
        dates: list[date],
        strategy: AllStrategies,
    ) -> list[TradeIntent]:
        """Generate trading intents."""
        intents: list[TradeIntent] = []
        for current_date in dates:
            signal = strategy.evaluate(
                current_date=current_date, market_snapshot=market_snapshot
            )
            intents.append(
                strategy.generate_intent(
                    current_date=current_date,
                    market_snapshot=market_snapshot,
                    signal=signal,
                )
            )

    def run(self, market_feed: MarketFeed, strategy: AllStrategies):
        """Run engine."""
        market_snapshot = market_feed.market_snapshot
        dates: list[date] = [
            _datetime for _datetime in market_feed.dates if _datetime >= self.start_date
        ]

        # Step 1: Generate all trade intents (once)
        intents = self.generate_trading_intents(
            market_snapshot=market_snapshot, dates=dates, strategy=strategy
        )

        # Step 2: Setup portfolio state
        portfolio = PortfolioState(cash=100_000.0)  # Starting capital
        executed_trades = []
        market_snapshot = market_feed.market_snapshot

        # Step 3: Main backtest loop
        for current_date in dates:
            # 3.1: Close any open positions if conditions met
            portfolio.evaluate_closures(current_date, market_snapshot)

            # 3.2: Get trade intents for the current date
            todays_intents = [
                intent for intent in intents if intent.intent_date == current_date
            ]

            # 3.3: Apply capital allocation / filters
            filtered_intents = portfolio.filter_trade_intents(todays_intents)

            # 3.4: Execute trades
            for intent in filtered_intents:
                trade = portfolio.execute_trade(intent, market_snapshot)
                if trade:
                    executed_trades.append(trade)

            # 3.5: Update portfolio state (e.g., mark-to-market)
            portfolio.update(current_date, market_snapshot)

        # Step 4: Post-backtest summary
        return BacktestResult(
            executed_trades=executed_trades,
            portfolio=portfolio,
            performance=portfolio.compute_metrics(),
        )
