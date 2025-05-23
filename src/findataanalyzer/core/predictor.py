"""Financial data prediction module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


class Predictor:
    """Financial data prediction class."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.models = {}
        self.scalers = {}
    
    def train(self, data: pd.DataFrame, target_column: str, 
              features: Optional[List[str]] = None, 
              method: str = "linear") -> Dict[str, Any]:
        """
        Train a prediction model on financial data.
        
        Args:
            data: DataFrame containing the financial data
            target_column: The column to predict
            features: List of feature columns to use for prediction
            method: Prediction method (linear, arima, etc.)
            
        Returns:
            Dictionary with training results
        """
        if method == "linear":
            return self._train_linear(data, target_column, features)
        elif method == "arima":
            return self._train_arima(data, target_column)
        else:
            raise ValueError(f"Unsupported prediction method: {method}")
    
    def predict(self, data: pd.DataFrame, target_column: str, 
                horizon: int = 1, method: str = "linear") -> Dict[str, Any]:
        """
        Make predictions on financial data.
        
        Args:
            data: DataFrame containing the financial data
            target_column: The column to predict
            horizon: Number of steps to predict into the future
            method: Prediction method (linear, arima, etc.)
            
        Returns:
            Dictionary with prediction results
        """
        if method == "linear":
            return self._predict_linear(data, target_column, horizon)
        elif method == "arima":
            return self._predict_arima(data, target_column, horizon)
        else:
            raise ValueError(f"Unsupported prediction method: {method}")
    
    def _train_linear(self, data: pd.DataFrame, target_column: str, 
                      features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train a linear regression model."""
        if features is None:
            # Use all numerical columns except the target as features
            features = [col for col in data.select_dtypes(include=['number']).columns 
                       if col != target_column]
        
        if not features:
            raise ValueError("No features available for prediction")
        
        # Prepare the data
        X = data[features].values
        y = data[target_column].values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Store the model and scaler
        self.models[target_column] = model
        self.scalers[target_column] = {
            'scaler': scaler,
            'features': features
        }
        
        # Calculate training metrics
        y_pred = model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        r2 = model.score(X_scaled, y)
        
        return {
            'method': 'linear',
            'target': target_column,
            'features': features,
            'metrics': {
                'mse': float(mse),
                'r2': float(r2)
            },
            'coefficients': {feat: float(coef) for feat, coef in zip(features, model.coef_)},
            'intercept': float(model.intercept_)
        }
    
    def _predict_linear(self, data: pd.DataFrame, target_column: str, 
                        horizon: int = 1) -> Dict[str, Any]:
        """Make predictions using a linear regression model."""
        if target_column not in self.models:
            raise ValueError(f"No trained model found for {target_column}. Call train() first.")
        
        model = self.models[target_column]
        scaler_info = self.scalers[target_column]
        scaler = scaler_info['scaler']
        features = scaler_info['features']
        
        # Prepare the data
        X = data[features].values
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # For future predictions (simple approach for demonstration)
        future_predictions = []
        last_data_point = data.iloc[-1:][features].values
        last_scaled = scaler.transform(last_data_point)
        
        next_pred = model.predict(last_scaled)[0]
        future_predictions.append(float(next_pred))
        
        for _ in range(1, horizon):
            # Very simplistic approach - in a real system, you would update features
            next_pred = next_pred * 1.01  # Just a placeholder
            future_predictions.append(float(next_pred))
        
        return {
            'method': 'linear',
            'target': target_column,
            'current_predictions': predictions.tolist(),
            'future_predictions': future_predictions
        }
    
    def _train_arima(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train an ARIMA model for time series prediction."""
        # Ensure data is sorted by time if there's a time column
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Get the target series
        y = data[target_column].values
        
        # Fit ARIMA model (simple configuration for demonstration)
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Store the model
        self.models[f"{target_column}_arima"] = model_fit
        
        # Get model summary
        summary = model_fit.summary()
        
        return {
            'method': 'arima',
            'target': target_column,
            'order': (1, 1, 1),
            'aic': float(model_fit.aic),
            'bic': float(model_fit.bic)
        }
    
    def _predict_arima(self, data: pd.DataFrame, target_column: str, 
                       horizon: int = 1) -> Dict[str, Any]:
        """Make predictions using an ARIMA model."""
        model_key = f"{target_column}_arima"
        if model_key not in self.models:
            raise ValueError(f"No trained ARIMA model found for {target_column}. Call train() first.")
        
        model_fit = self.models[model_key]
        
        # Make predictions
        forecast = model_fit.forecast(steps=horizon)
        
        return {
            'method': 'arima',
            'target': target_column,
            'forecast': forecast.tolist()
        } 