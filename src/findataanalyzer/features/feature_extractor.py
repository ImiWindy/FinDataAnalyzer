"""
Module for feature extraction from multi-timeframe market data.
"""
import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts features from multi-timeframe data to be used by an ML model.
    """

    def __init__(self, config: Dict = None):
        """
        Initializes the FeatureExtractor.

        Args:
            config: Configuration dictionary, e.g., for indicator parameters.
        """
        self.config = config or {}
        logger.info("FeatureExtractor initialized.")

    def calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculates technical indicators for a given DataFrame.

        Args:
            data: A pandas DataFrame with OHLCV data.
            timeframe: The timeframe of the data (e.g., '1h', '15m') for unique column naming.

        Returns:
            The original DataFrame with new indicator columns.
        """
        # SMA - Simple Moving Average
        sma_short_period = self.config.get('sma_short', 10)
        sma_long_period = self.config.get('sma_long', 30)
        data[f'sma_short_{timeframe}'] = data['close'].rolling(window=sma_short_period).mean()
        data[f'sma_long_{timeframe}'] = data['close'].rolling(window=sma_long_period).mean()

        # RSI - Relative Strength Index
        rsi_period = self.config.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data[f'rsi_{timeframe}'] = 100 - (100 / (1 + rs))

        return data

    def generate_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generates a combined feature set from multi-timeframe data.

        Args:
            market_data: A dictionary mapping timeframes to OHLCV DataFrames.

        Returns:
            A single DataFrame with features from all timeframes, aligned by timestamp.
        """
        if not market_data:
            logger.warning("Market data is empty. Cannot generate features.")
            return pd.DataFrame()

        feature_dfs = []
        for timeframe, df in market_data.items():
            if df.empty:
                continue
            
            # Ensure the index is a DatetimeIndex
            df.index = pd.to_datetime(df.index)

            # Calculate indicators for the current timeframe
            features = self.calculate_indicators(df.copy(), timeframe)
            feature_dfs.append(features)

        if not feature_dfs:
            return pd.DataFrame()

        # Combine features from all timeframes
        # We start with the highest frequency data as the base (assuming it's the first in the dict)
        base_df = feature_dfs[0].copy()
        
        # Use 'ffill' to propagate values from slower timeframes to faster ones
        for other_df in feature_dfs[1:]:
            base_df = pd.merge(base_df, other_df, on='timestamp', how='left', suffixes=('', '_y'))
            base_df.drop(base_df.filter(regex='_y$').columns, axis=1, inplace=True)

        # Forward fill any NaNs that result from merging different frequencies
        base_df.fillna(method='ffill', inplace=True)
        
        # Drop initial rows with NaNs from rolling calculations
        base_df.dropna(inplace=True)

        logger.info("Generated combined feature DataFrame with shape: %s", base_df.shape)
        return base_df 