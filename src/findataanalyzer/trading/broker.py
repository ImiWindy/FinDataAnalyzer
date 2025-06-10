"""Broker interface for trading."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import alpaca_trade_api as tradeapi

from findataanalyzer.core.config import settings

class Broker(ABC):
    """Abstract base class for a trading broker."""

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_class: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a new order."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a given symbol."""
        pass

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get the latest quote for a given symbol."""
        pass

class AlpacaBroker(Broker):
    """Implementation of a broker for Alpaca."""

    def __init__(
        self,
        api_key: str = settings.ALPACA_API_KEY,
        api_secret: str = settings.ALPACA_API_SECRET,
        base_url: str = settings.ALPACA_PAPER_TRADING_URL
    ):
        """
        Initialize the Alpaca broker.

        Args:
            api_key: Your Alpaca API key.
            api_secret: Your Alpaca API secret.
            base_url: The base URL for the Alpaca API.
        """
        if not api_key or not api_secret:
            raise ValueError("Alpaca API key and secret must be set.")
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        account = self.api.get_account()
        return account._raw

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_class: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order on Alpaca.

        Args:
            symbol: The stock symbol to trade.
            qty: The number of shares to trade.
            side: 'buy' or 'sell'.
            order_type: 'market', 'limit', 'stop', 'stop_limit'.
            time_in_force: 'day', 'gtc', 'opg'.
            limit_price: The limit price for limit orders.
            stop_price: The stop price for stop orders.
            order_class: 'simple' or 'bracket' for advanced orders.
            take_profit_price: The take profit price for bracket orders.
            stop_loss_price: The stop loss price for bracket orders.
        
        Returns:
            A dictionary representing the placed order.
        """
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "limit_price": limit_price,
            "stop_price": stop_price,
        }

        if order_class == 'bracket':
            if not take_profit_price or not stop_loss_price:
                raise ValueError("take_profit_price and stop_loss_price are required for bracket orders.")
            order_data["order_class"] = "bracket"
            order_data["take_profit"] = {"limit_price": take_profit_price}
            order_data["stop_loss"] = {"stop_price": stop_loss_price}

        # Filter out None values so they don't get passed to the API
        order_data = {k: v for k, v in order_data.items() if v is not None}

        order = self.api.submit_order(**order_data)
        return order._raw

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a given symbol on Alpaca."""
        try:
            position = self.api.get_position(symbol)
            return position._raw
        except tradeapi.rest.APIError as e:
            if e.status_code == 404:
                return {}  # No position found
            raise e

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get the latest quote for a given symbol from Alpaca."""
        quote = self.api.get_latest_quote(symbol)
        return quote._raw 