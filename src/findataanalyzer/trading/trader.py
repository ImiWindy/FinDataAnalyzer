"""The core trading engine."""

from typing import Dict, Any, List
import logging

from findataanalyzer.trading.broker import Broker
from findataanalyzer.trading.risk_manager import RiskManager
from findataanalyzer.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

# Let's define a more structured signal, as you suggested.
# This would typically be in its own file, e.g., types.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TradeSignal:
    """A structured trading signal."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    position_size_pct: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    rationale: str = ""


class Trader:
    """
    The Trader class orchestrates the trading process.
    
    It takes a strategy and a broker, fetches market data,
    gets signals from the strategy, and places orders via the broker.
    """

    def __init__(self, broker: Broker, strategy: BaseStrategy, risk_manager: RiskManager, config: Dict[str, Any]):
        """
        Initializes the Trader.
        
        Args:
            broker: An instance of a Broker implementation.
            strategy: An instance of a BaseStrategy implementation.
            risk_manager: An instance of the RiskManager.
            config: A dictionary for trader-specific configurations.
        """
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config
        logger.info("Trader initialized with broker %s, strategy %s, and risk_manager %s", 
                    broker.__class__.__name__, 
                    strategy.__class__.__name__,
                    risk_manager.__class__.__name__)

    def _fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetches market data for the given symbols.
        
        NOTE: This is a placeholder. A real implementation would fetch data
        from an exchange or data provider API into a structured format like a DataFrame.
        """
        logger.info("Fetching market data for symbols: %s", symbols)
        # Placeholder data - in a real scenario, this would call the broker
        import pandas as pd
        return {symbol: pd.DataFrame() for symbol in symbols}

    def run_trading_cycle(self, external_data: Dict[str, pd.DataFrame] = None):
        """
        Executes one full cycle of the trading logic.
        
        Args:
            external_data: Optional. A dictionary mapping symbols to pandas DataFrames.
                           If provided, this data is used instead of fetching live data.
                           This is useful for backtesting.
        """
        logger.info("Starting new trading cycle...")
        
        symbols_to_trade = self.config.get("symbols", [])
        if not symbols_to_trade:
            logger.warning("No symbols configured for trading. Ending cycle.")
            return

        if external_data:
            logger.info("Using external data for trading cycle.")
            market_data = external_data
        else:
            logger.info("Fetching live market data.")
            market_data = self._fetch_market_data(symbols_to_trade)

        for symbol in symbols_to_trade:
            data_for_symbol = market_data.get(symbol)
            if data_for_symbol is None or data_for_symbol.empty:
                logger.warning("No data found for symbol %s in this cycle.", symbol)
                continue
                
            # The agent/strategy should return a list of TradeSignal objects
            signals: List[TradeSignal] = self.strategy.generate_signals(data_for_symbol)
            
            self._process_signals(signals)

        logger.info("Trading cycle finished.")

    def _process_signals(self, signals: List[TradeSignal]):
        """Processes the generated signals and places orders with risk management."""
        if not signals:
            return
            
        try:
            account_info = self.broker.get_account_info()
            equity = float(account_info.get("equity", 0))
        except Exception as e:
            logger.error("Failed to get account info: %s. Cannot process signals.", e)
            return

        if equity == 0:
            logger.error("Account equity is zero. Cannot place trades.")
            return

        for signal in signals:
            logger.info("Processing signal: %s", signal)
            
            if signal.action == 'BUY':
                self._execute_buy_signal(signal, equity)
            elif signal.action == 'SELL':
                self._execute_sell_signal(signal, equity)
            elif signal.action == 'HOLD':
                logger.info("Signal is HOLD for %s. No action taken.", signal.symbol)

    def _execute_buy_signal(self, signal: TradeSignal, equity: float):
        """Executes a buy order with full risk management."""
        if not all([signal.stop_loss_price, signal.take_profit_price]):
            logger.warning("BUY signal for %s is missing stop_loss or take_profit. Cannot place bracket order.", signal.symbol)
            return

        try:
            quote = self.broker.get_latest_quote(signal.symbol)
            # Use ask price for buy orders as a conservative estimate
            entry_price = quote.get('ap') 
            if not entry_price:
                logger.error("Could not get a valid ask price for %s.", signal.symbol)
                return

            quantity = self.risk_manager.calculate_position_size(
                account_equity=equity,
                entry_price=entry_price,
                stop_loss_price=signal.stop_loss_price,
                risk_percentage=signal.position_size_pct # can be None
            )

            if quantity <= 0:
                logger.warning("Calculated quantity for %s is zero or less. Skipping trade.", signal.symbol)
                return

            order = self.broker.place_order(
                symbol=signal.symbol,
                qty=quantity,
                side='buy',
                order_type='market',
                time_in_force='gtc', # Good 'til Canceled for bracket orders
                order_class='bracket',
                take_profit_price=signal.take_profit_price,
                stop_loss_price=signal.stop_loss_price,
            )
            logger.info("Placed BUY bracket order for %s. Order details: %s", signal.symbol, order)
            logger.info("REASON: %s", signal.rationale)

        except Exception as e:
            logger.error("Failed to place BUY order for %s: %s", signal.symbol, e)

    def _execute_sell_signal(self, signal: TradeSignal, equity: float):
        """Executes a sell order. Can be for closing a position."""
        # For now, implementing a simple market sell of a specified quantity.
        # A more advanced version would check current position.
        
        # We need to know the size of the position to sell.
        # This part requires more logic: is it a short-sell or closing a long?
        # Assuming for now it's closing a long position fully.
        try:
            position = self.broker.get_position(signal.symbol)
            if not position:
                logger.warning("Received SELL signal for %s but no position found.", signal.symbol)
                return
            
            quantity_to_sell = float(position['qty'])
            
            order = self.broker.place_order(
                symbol=signal.symbol,
                qty=quantity_to_sell,
                side='sell',
                order_type='market',
                time_in_force='day'
            )
            logger.info("Placed SELL order to close position in %s. Order: %s", signal.symbol, order)
            logger.info("REASON: %s", signal.rationale)

        except Exception as e:
            logger.error("Failed to place SELL order for %s: %s", signal.symbol, e) 