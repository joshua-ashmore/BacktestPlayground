"""Interactive Brokers Execution Engine using IBAPI."""

from collections import defaultdict
from datetime import date
from typing import List, Literal, Optional, Tuple

from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper
from pydantic import ConfigDict, model_validator

from components.execution.base_model import ExecutionEngineBase
from components.job.base_model import StrategyJob
from components.strategies.base_model import Strategy
from components.trades.intent_model import TradeIntent
from components.trades.trade_model import Trade, TradeLeg


class IBApp(EWrapper, EClient):
    """Thin IBAPI client wrapper for synchronous calls."""

    def __init__(self):
        EClient.__init__(self, self)
        self.next_order_id: Optional[int] = None
        self.market_data = {}
        self.account_values = {}
        self.contract_details = {}

    def nextValidId(self, orderId: int):
        self.next_order_id = orderId

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        self.market_data[reqId] = price

    def updateAccountValue(self, key, val, currency, accountName):
        self.account_values[key] = (val, currency)

    def contractDetails(self, reqId, contractDetails):
        self.contract_details[reqId] = contractDetails


class IBExecutionEngine(ExecutionEngineBase):
    """IB Execution Engine Model."""

    engine_type: Literal["ib"] = "ib"

    cash: float = 0

    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # TWS: 7496, IB Gateway Paper: 7497
    client_id: int = 1

    app: Optional[IBApp] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.app = IBApp()
        self.app.connect(self.ib_host, self.ib_port, self.client_id)
        self.app.next_order_id = None
        self._wait_for_next_order_id()

        # Fetch account cash balance
        self._request_account_values()
        self.cash = float(self.app.account_values.get("TotalCashValue", (0, "USD"))[0])
        if self.positions is None:
            self.positions = defaultdict(float)
        return self

    def _wait_for_next_order_id(self):
        """Wait until IB API sends nextValidId."""
        import time

        start = time.time()
        while self.app.next_order_id is None and (time.time() - start < 5):
            self.app.run()

    def _request_account_values(self):
        self.app.reqAccountSummary(9001, "All", "TotalCashValue")
        import time

        time.sleep(1)  # crude wait; replace with condition variable in production

    def _create_contract(
        self, symbol: str, sec_type="STK", currency="USD", exchange="SMART"
    ):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        contract.exchange = exchange
        return contract

    def _create_order(self, action: str, quantity: float, order_type="MKT"):
        order = Order()
        order.action = action
        order.orderType = order_type
        order.totalQuantity = quantity
        order.transmit = True
        return order

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Request real-time market price from IB."""
        contract = self._create_contract(symbol)
        req_id = self.app.next_order_id
        self.app.reqMktData(req_id, contract, "", False, False, [])
        import time

        time.sleep(1)
        return self.app.market_data.get(req_id)

    # def _open_trade(
    #     self,
    #     job: StrategyJob,
    #     intent: TradeIntent,
    #     symbol: Tuple[str, ...],
    # ) -> List[Trade]:
    #     """Open a new trade with IB."""
    #     trade_legs = []
    #     # TODO: add logic to ensure that all orders fill for legged trades or none
    #     for intent_leg in intent.legs:
    # price = self._get_market_price(symbol=intent_leg.symbol)
    # if price is None:
    #     return []

    #         notional = intent_leg.weight * self.cash * self.allocation_pct_per_trade
    #         quantity = round(notional / price, 0)

    #         # Send order to IB
    #         contract = self._create_contract(intent_leg.symbol)
    #         order = self._create_order(intent_leg.signal.upper(), quantity)
    #         self.app.placeOrder(self.app.next_order_id, contract, order)
    #         self.app.next_order_id += 1

    #         trade_legs.append(
    #             TradeLeg(
    #                 symbol=intent_leg.symbol,
    #                 open_date=intent_leg.date,
    #                 quantity=quantity,
    #                 price=price,
    #                 notional=notional,
    #                 strategy=intent_leg.strategy,
    #                 side=intent_leg.signal,
    #             )
    #         )

    #     if not trade_legs:
    #         return []

    #     trade = Trade(legs=trade_legs, metadata=intent.metadata)
    #     self.open_positions[symbol] = trade
    #     self.cash -= sum([leg.notional for leg in trade.legs])
    #     return [trade]

    def _open_trade(
        self,
        job: StrategyJob,
        intent: TradeIntent,
        symbol: Tuple[str, ...],
    ) -> List[Trade]:
        """Open new trade with parent-child order grouping."""

        trade_legs: List[TradeLeg] = []

        # 1. Get the next order id for parent
        parent_order_id = self.app.next_order_id
        self.app.next_order_id += 1

        # 2. Create a dummy parent contract (can be first leg's contract or a generic contract)
        parent_contract = self._create_contract(intent.legs[0].symbol)

        # 3. Create the parent order — zero quantity, just to group children
        parent_order = Order()
        parent_order.orderId = parent_order_id
        parent_order.action = "BUY"  # or any valid action; won't be executed
        parent_order.orderType = "MKT"
        parent_order.totalQuantity = 0  # zero quantity — parent is dummy
        parent_order.transmit = False  # do not send yet, wait for children

        self.app.placeOrder(
            orderId=parent_order_id, contract=parent_contract, order=parent_order
        )

        # 4. Create all child orders tied to the parent order
        for idx, intent_leg in enumerate(intent.legs):
            price = self._get_market_price(symbol=intent_leg.symbol)
            if price is None:
                return []

            notional = intent_leg.weight * self.cash * self.allocation_pct_per_trade
            quantity = round(notional / price, 0)

            contract = self._create_contract(intent_leg.symbol)
            child_order = Order()
            child_order.orderId = self.app.next_order_id
            self.app.next_order_id += 1

            child_order.action = intent_leg.signal.upper()
            child_order.orderType = "MKT"
            child_order.totalQuantity = quantity
            child_order.parentId = parent_order_id  # tie to parent order

            # Only the last child order has transmit=True to send entire batch
            child_order.transmit = idx == len(intent.legs) - 1

            self.app.placeOrder(child_order.orderId, contract, child_order)

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

        # 5. Update cash balance after order submission
        self.cash -= sum([leg.notional for leg in trade_legs])

        # 6. Save open position locally
        trade = Trade(legs=trade_legs, metadata=intent.metadata)
        self.open_positions[symbol] = trade

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
        """Close existing trade with IB."""
        close_signal = strategy.generate_exit_signals(
            job=job, trade=existing_trade, current_date=target_date
        )
        trade_age = (intent.date - existing_trade.legs[0].open_date).days

        if trade_age >= self.max_hold_days or close_signal:
            for leg in existing_trade.legs:
                price = self._get_market_price(symbol=leg.symbol)
                if price is None:
                    return

                # Send close order
                action = "SELL" if leg.side.upper() == "BUY" else "BUY"
                contract = self._create_contract(leg.symbol)
                order = self._create_order(action, leg.quantity)
                self.app.placeOrder(self.app.next_order_id, contract, order)
                self.app.next_order_id += 1

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
                price = self._get_market_price(symbol=leg.symbol)
                if price is None:
                    return
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
                    price = self._get_market_price(symbol=leg.symbol)
                    if price is None:
                        return
                    leg.close_date = target_date
                    leg.close_price = price
                    leg.pnl = (
                        (price - leg.price) * leg.quantity * leg.direction_multiplier
                    )
                    self.cash += leg.pnl + leg.notional
                del self.open_positions[symbol]
