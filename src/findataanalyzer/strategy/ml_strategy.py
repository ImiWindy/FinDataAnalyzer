"""
A trading strategy that uses a Feature Extractor and an ML Predictor
to generate trading signals from multi-timeframe data.
"""
import logging
from typing import Any, Dict, List
import pandas as pd
from pathlib import Path

from findataanalyzer.strategy.base import BaseStrategy, TradeSignal
from findataanalyzer.features.feature_extractor import FeatureExtractor
from findataanalyzer.core.predictor import Predictor

logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    """
    Implements a machine learning-based trading strategy.
    
    This strategy works in a sequence:
    1. Extracts multi-timeframe features using a FeatureExtractor.
    2. Checks for a specific trigger condition (e.g., a crossover).
    3. If triggered, it uses a Predictor model to get a success probability.
    4. If the probability is above a threshold, it generates a trade signal.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MLStrategy.

        Args:
            config: A dictionary containing strategy-specific parameters.
                    Expected keys:
                    - 'model_path': Path to the trained ML model.
                    - 'prediction_threshold': Probability threshold to trigger a trade.
                    - 'risk_reward_ratio': For calculating take-profit.
        """
        super().__init__(config)
        
        self.feature_extractor = FeatureExtractor(config=self.config.get("feature_config", {}))
        
        model_path_str = self.config.get("model_path")
        if not model_path_str:
            raise ValueError("model_path must be specified in the strategy config.")
        
        self.predictor = Predictor(model_path=Path(model_path_str))
        self.threshold = self.config.get("prediction_threshold", 0.60)
        self.risk_reward_ratio = self.config.get("risk_reward_ratio", 1.5)
        
        # New setup/trigger timeframe configuration
        self.setup_timeframes = self.config.get("setup_timeframes", ['1h', '15m'])
        self.trigger_timeframe = self.config.get("trigger_timeframe", '5m')

        logger.info(
            "MLStrategy initialized with setup TFs: %s, trigger TF: %s, threshold: %.2f",
            self.setup_timeframes, self.trigger_timeframe, self.threshold
        )

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """
        Generates trading signals using the ML model pipeline with setup/trigger logic.
        
        Args:
            market_data: A dictionary mapping timeframes to OHLCV DataFrames.
                         
        Returns:
            A list containing at most one TradeSignal.
        """
        if self.trigger_timeframe not in market_data:
            logger.warning("Trigger timeframe '%s' not found in market data.", self.trigger_timeframe)
            return []
            
        # 1. Extract features from setup timeframes
        setup_data = {tf: df for tf, df in market_data.items() if tf in self.setup_timeframes}
        features_df = self.feature_extractor.generate_features(setup_data)
        if features_df.empty:
            return [] # Not enough data for features
        
        latest_features = features_df.iloc[-1]
        
        # 2. Check for a trigger condition on the trigger timeframe
        trigger_df = market_data[self.trigger_timeframe]
        latest_trigger_candle = trigger_df.iloc[-1]
        
        # Example trigger: Price must close above the 20-period SMA on the trigger timeframe
        trigger_sma = latest_trigger_candle['close'].rolling(window=20).mean().iloc[-1]
        if latest_trigger_candle['close'] <= trigger_sma:
            return [] # Trigger condition not met

        logger.info("Trigger condition MET on %s timeframe.", self.trigger_timeframe)

        # 3. Get prediction from the model using features from setup timeframes
        prediction_input = latest_features.to_frame().T
        # Add the latest close from trigger timeframe as a feature
        prediction_input['close'] = latest_trigger_candle['close']
        
        success_prob_array = self.predictor.predict_proba(prediction_input)

        if success_prob_array.size == 0:
            logger.error("Prediction failed. Cannot generate signal.")
            return []
            
        success_prob = success_prob_array[0]
        logger.info("Prediction received. Success probability: %.4f", success_prob)

        # 4. Generate signal if probability is above the threshold
        if success_prob >= self.threshold:
            logger.info("Probability %.4f is >= threshold %.2f. Generating BUY signal.", success_prob, self.threshold)
            
            entry_price = latest_trigger_candle['close']
            stop_loss_price = latest_trigger_candle['low'] * 0.995 # Tighter stop loss based on trigger candle
            risk_per_share = entry_price - stop_loss_price
            
            if risk_per_share <= 0:
                logger.warning("Invalid risk per share calculation. Cannot create signal.")
                return []

            take_profit_price = entry_price + (risk_per_share * self.risk_reward_ratio)

            signal = TradeSignal(
                symbol=self.config.get("symbol", "UNKNOWN"),
                action='BUY',
                entry_price=round(entry_price, 2),
                stop_loss_price=round(stop_loss_price, 2),
                take_profit_price=round(take_profit_price, 2),
                predicted_success_prob=round(success_prob, 4),
                rationale=(
                    f"ML signal. Trigger: Close > SMA(20) on {self.trigger_timeframe}. "
                    f"Prob ({success_prob:.2f}) > Threshold ({self.threshold:.2f})."
                )
            )
            return [signal]
        
        return [] 