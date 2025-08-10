"""Interactive Brokers Data Source."""

import threading
import time
from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from ibapi.client import EClient
from ibapi.common import BarData as IBBarData
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper
from pydantic import BaseModel, ConfigDict, model_validator

from components.market.base import BarData


class IBApiClient(EWrapper, EClient):
    """Interactive Brokers API Client."""

    def __init__(self):
        """Init."""
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.req_id_counter = 0
        self.data = {}
        self.reqId_to_symbol = {}
        self.next_valid_order_id = None
        self.connected = False

        self.data_events = {}
        self.lock = threading.Lock()

    # ---------- ID MANAGEMENT ----------
    def get_next_req_id(self) -> int:
        with self.lock:
            self.req_id_counter += 1
            return self.req_id_counter

    # ---------- CONTRACT CREATION ----------
    def create_contract(
        self, symbol: str, sec_type="STK", exchange="SMART", currency="USD"
    ) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    # ---------- WAIT FOR DATA ----------
    def wait_for_data(self, req_id: int, timeout: int = 10) -> bool:
        """
        Wait for the Event associated with this req_id to be set by historicalDataEnd.
        """
        event = self.data_events.get(req_id)
        if event:
            return event.wait(timeout=timeout)
        return False

    # ---------- HISTORICAL DATA CALLBACK ----------
    def historicalData(self, reqId: int, bar: IBBarData):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        # Signal that this request is done
        if reqId in self.data_events:
            self.data_events[reqId].set()

    # ---------- FETCH DATA ----------
    def fetch_data(self, symbol: str, missing_dates: list) -> list[BarData]:
        contract = self.create_contract(symbol)

        if missing_dates:
            min_date = min(missing_dates)
            max_date = max(missing_dates)
            delta_days = (max_date - min_date).days + 1
            duration_str = f"{delta_days} D"
        else:
            duration_str = "1 M"

        req_id = self.get_next_req_id()
        self.data_events[req_id] = threading.Event()

        self.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        if not self.wait_for_data(req_id):
            print(f"Timeout waiting for data for {symbol}")
            return []

        raw_bars = self.data.get(req_id, [])
        return [
            BarData(
                datetime=datetime.strptime(bar.date, "%Y%m%d").date(),
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            for bar in raw_bars
        ]


class IBDataSource(BaseModel):
    """Interactive Brokers Data Source."""

    data_source: Literal["ib"] = "ib"

    host: str = "127.0.0.1"
    port: int = 7479
    client_id: int = 1

    client: Optional[IBApiClient] = None
    api_thread: Optional[threading.Thread] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def post_validate_model(self):
        """Post validate model."""
        self.client = IBApiClient()
        self.client.connect(host=self.host, port=self.port, clientId=self.client_id)
        self.api_thread = threading.Thread(target=self.client.run, daemon=True)
        self.api_thread.start()

        while not self.client.connected:
            print("[IBPaperTradingFeed] Waiting for connection...")
            time.sleep(0.1)

    def fetch_data(
        self,
        symbol: str,
        missing_dates: list[date],
    ) -> List[Dict]:
        """Fetch data from Interactive Brokers as a list of dict."""
        bar_list = self.client.fetch_data(symbol=symbol, missing_dates=missing_dates)
        return [bar.model_dump() for bar in bar_list]

    def request_historical_data(self, symbol: str, duration="1 M", bar_size="1 day"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        req_id = self.client.nextValidId()
        self.client.reqId_to_symbol[req_id] = symbol
        self.client.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )
        print(f"[IBPaperTradingFeed] Requested historical data for {symbol}")

    def place_market_order(self, symbol: str, action: str, quantity: int):
        if self.client.next_valid_order_id is None:
            raise RuntimeError("Order ID not ready")

        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity

        order_id = self.client.next_valid_order_id
        print(
            f"[IBPaperTradingFeed] Placing order id={order_id}: {action} {quantity} {symbol}"
        )
        self.client.placeOrder(order_id, contract, order)
        self.client.next_valid_order_id += 1

    def disconnect(self):
        self.client.disconnect()
        print("[IBPaperTradingFeed] Disconnected from IB API")


# if __name__ == "__main__":
#     feed = IBPaperTradingFeed()

#     # Request 1 month daily bars for AAPL
#     feed.request_historical_data("AAPL", req_id=1, duration="1 M", bar_size="1 day")

#     # Wait to receive data callbacks
#     time.sleep(10)

#     # Place a paper market buy order for 1 share of AAPL
#     feed.place_market_order("AAPL", "BUY", 1)

#     # Give some time for order to be placed and callbacks to process
#     time.sleep(5)

#     feed.disconnect()
