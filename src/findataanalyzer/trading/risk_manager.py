"""
Risk management module for calculating position sizes and other risk-related metrics.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Handles risk management, including position sizing calculations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RiskManager.

        Args:
            config: A dictionary for risk-management-specific configurations.
                    Example: {"default_risk_per_trade_pct": 1.0}
        """
        self.config = config
        self.default_risk_pct = self.config.get("default_risk_per_trade_pct", 1.0)
        logger.info("RiskManager initialized with config: %s", config)

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percentage: float = None
    ) -> float:
        """
        Calculates the position size in shares/units.

        Args:
            account_equity: The total equity of the trading account.
            entry_price: The estimated entry price for the trade.
            stop_loss_price: The price at which to exit for a loss.
            risk_percentage: The percentage of account equity to risk on this trade.
                               If None, uses the default from the config.

        Returns:
            The number of shares/units to purchase. Returns 0 if risk is invalid.
        """
        if risk_percentage is None:
            risk_percentage = self.default_risk_pct

        if entry_price <= stop_loss_price:
            logger.warning("Stop loss price must be below entry price for a long trade.")
            return 0.0

        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0.0
            
        amount_to_risk = account_equity * (risk_percentage / 100.0)
        
        position_size = amount_to_risk / risk_per_share
        
        logger.info(
            "Calculated position size: %.2f shares for a %.2f%% risk on equity of %.2f",
            position_size, risk_percentage, account_equity
        )
        return round(position_size, 2) # Assuming fractional shares are allowed 