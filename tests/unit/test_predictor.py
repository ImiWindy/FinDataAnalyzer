"""Tests for the predictor module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from findataanalyzer.core.predictor import Predictor


class TestPredictor:
    """Tests for the Predictor class."""
    
    def test_init(self):
        """Test initialization of the Predictor."""
        predictor = Predictor()
        assert hasattr(predictor, "models")
        assert hasattr(predictor, "scalers")
        assert predictor.models == {}
        assert predictor.scalers == {}
    
    def test_train_invalid_method(self, sample_data):
        """Test training with invalid method."""
        predictor = Predictor()
        with pytest.raises(ValueError, match="Unsupported prediction method"):
            predictor.train(sample_data, "price", method="invalid_method")
    
    def test_predict_invalid_method(self, sample_data):
        """Test prediction with invalid method."""
        predictor = Predictor()
        with pytest.raises(ValueError, match="Unsupported prediction method"):
            predictor.predict(sample_data, "price", method="invalid_method")
    
    @patch("findataanalyzer.core.predictor.LinearRegression")
    @patch("findataanalyzer.core.predictor.StandardScaler")
    def test_train_linear(self, mock_scaler_class, mock_lr_class, sample_data):
        """Test the _train_linear method."""
        # Setup mocks
        mock_scaler = MagicMock()
        mock_scaler_class.return_value = mock_scaler
        mock_scaler.fit_transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])
        
        mock_lr = MagicMock()
        mock_lr_class.return_value = mock_lr
        mock_lr.coef_ = np.array([0.5, 1.5])
        mock_lr.intercept_ = 0.1
        mock_lr.score.return_value = 0.95
        mock_lr.predict.return_value = np.array([101, 102, 103])
        
        # Create predictor and call train
        predictor = Predictor()
        result = predictor._train_linear(
            sample_data, "price", features=["volume"]
        )
        
        # Verify calls
        mock_scaler.fit_transform.assert_called_once()
        mock_lr.fit.assert_called_once()
        
        # Verify result
        assert result["method"] == "linear"
        assert result["target"] == "price"
        assert result["features"] == ["volume"]
        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert "mse" in result["metrics"]
        assert "coefficients" in result
        assert "intercept" in result
        
        # Verify model and scaler are stored
        assert "price" in predictor.models
        assert "price" in predictor.scalers
    
    def test_train_linear_no_features(self, sample_data):
        """Test _train_linear with no features specified."""
        predictor = Predictor()
        result = predictor._train_linear(sample_data, "price")
        
        assert result["method"] == "linear"
        assert len(result["features"]) == 1
        assert "volume" in result["features"]
    
    def test_predict_linear_no_model(self, sample_data):
        """Test _predict_linear with no trained model."""
        predictor = Predictor()
        with pytest.raises(ValueError, match="No trained model found"):
            predictor._predict_linear(sample_data, "price")
    
    def test_predict_linear(self, sample_data):
        """Test the _predict_linear method."""
        # Setup a trained model
        predictor = Predictor()
        
        # Mock the model and scaler
        predictor.models["price"] = MagicMock()
        predictor.models["price"].predict.return_value = np.array([105, 106, 107])
        
        predictor.scalers["price"] = {
            "scaler": MagicMock(),
            "features": ["volume"]
        }
        predictor.scalers["price"]["scaler"].transform.return_value = np.array([[1], [2], [3]])
        
        # Call predict
        result = predictor._predict_linear(sample_data, "price", horizon=2)
        
        # Verify calls
        predictor.scalers["price"]["scaler"].transform.assert_called()
        predictor.models["price"].predict.assert_called()
        
        # Verify result
        assert result["method"] == "linear"
        assert result["target"] == "price"
        assert "current_predictions" in result
        assert "future_predictions" in result
        assert len(result["future_predictions"]) == 2
    
    @patch("findataanalyzer.core.predictor.ARIMA")
    def test_train_arima(self, mock_arima_class, sample_data):
        """Test the _train_arima method."""
        # Setup mocks
        mock_arima = MagicMock()
        mock_arima_class.return_value = mock_arima
        
        mock_model_fit = MagicMock()
        mock_arima.fit.return_value = mock_model_fit
        mock_model_fit.aic = 123.45
        mock_model_fit.bic = 234.56
        
        # Call train_arima
        predictor = Predictor()
        result = predictor._train_arima(sample_data, "price")
        
        # Verify calls
        mock_arima.fit.assert_called_once()
        
        # Verify result
        assert result["method"] == "arima"
        assert result["target"] == "price"
        assert result["order"] == (1, 1, 1)
        assert result["aic"] == 123.45
        assert result["bic"] == 234.56
        
        # Verify model is stored
        assert "price_arima" in predictor.models
    
    def test_predict_arima_no_model(self, sample_data):
        """Test _predict_arima with no trained model."""
        predictor = Predictor()
        with pytest.raises(ValueError, match="No trained ARIMA model found"):
            predictor._predict_arima(sample_data, "price")
    
    def test_predict_arima(self, sample_data):
        """Test the _predict_arima method."""
        # Setup a trained model
        predictor = Predictor()
        
        # Mock the model
        predictor.models["price_arima"] = MagicMock()
        predictor.models["price_arima"].forecast.return_value = np.array([105, 106])
        
        # Call predict
        result = predictor._predict_arima(sample_data, "price", horizon=2)
        
        # Verify calls
        predictor.models["price_arima"].forecast.assert_called_with(steps=2)
        
        # Verify result
        assert result["method"] == "arima"
        assert result["target"] == "price"
        assert "forecast" in result
        assert len(result["forecast"]) == 2 