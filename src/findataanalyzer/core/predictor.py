"""
Module for loading a pre-trained model and making predictions.
"""
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class Predictor:
    """
    A wrapper for a pre-trained scikit-learn model to make predictions.
    """

    def __init__(self, model_path: Path):
        """
        Initializes the Predictor by loading the model from the given path.

        Args:
            model_path: The path to the saved model file (e.g., .joblib).
        
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the model from the specified path."""
        try:
            logger.info("Loading model from %s", self.model_path)
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error("Model file not found at %s.", self.model_path)
            raise
        except Exception as e:
            logger.error("An error occurred while loading the model: %s", e)
            raise

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predicts the success probability for a given set of features.

        Args:
            features: A pandas DataFrame where columns match the features
                      the model was trained on.

        Returns:
            A numpy array of prediction probabilities for the positive class (class 1).
            Returns an empty array if prediction fails.
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot make predictions.")
            return np.array([])
            
        if not isinstance(features, pd.DataFrame) or features.empty:
            logger.warning("Input features are empty or not a DataFrame. No prediction.")
            return np.array([])

        try:
            # Reorder columns to match model's expected feature order, if available
            if hasattr(self.model, 'feature_names_in_'):
                model_features = self.model.feature_names_in_
                features = features[model_features]

            # The output of predict_proba is [prob_class_0, prob_class_1]
            # We are interested in the probability of the positive class (1)
            probabilities = self.model.predict_proba(features)
            
            # Return the probability of the second class (index 1)
            return probabilities[:, 1]
            
        except KeyError as e:
            logger.error("Prediction failed. A required feature is missing from the input: %s", e)
            return np.array([])
        except Exception as e:
            logger.error("An error occurred during prediction: %s", e)
            return np.array([])

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This assumes you have run `create_test_model.py` first
    MODEL_FILE = Path("models/test_tabular_model.joblib")
    
    if not MODEL_FILE.exists():
        print("Model file not found. Please run `create_test_model.py` first.")
    else:
        # Create dummy data for prediction
        dummy_data = {
            'sma_short_15m': [102], 'sma_long_15m': [100], 'rsi_15m': [60],
            'sma_short_1h': [105], 'rsi_1h': [65]
        }
        feature_df = pd.DataFrame(dummy_data)
        
        print("Attempting to predict with dummy data:")
        print(feature_df)
        
        predictor = Predictor(model_path=MODEL_FILE)
        success_prob = predictor.predict_proba(feature_df)
        
        if success_prob.size > 0:
            print(f"\nPredicted success probability: {success_prob[0]:.4f}")
        else:
            print("\nPrediction failed.") 