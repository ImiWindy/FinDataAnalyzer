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
    ) -> Dict[str, Any]:
        """Place a new order."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a given symbol."""
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
        
        Returns:
            A dictionary representing the placed order.
        """
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
        )
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