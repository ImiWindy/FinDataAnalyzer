"""Integration tests for data flow from loading to analysis and prediction."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from findataanalyzer.core.data_loader import DataLoader
from findataanalyzer.core.analyzer import DataAnalyzer
from findataanalyzer.core.predictor import Predictor


class TestDataFlow:
    """Integration tests for the data flow."""
    
    def test_load_analyze_predict_flow(self, sample_ohlc_data):
        """Test the full flow from loading data to analysis to prediction."""
        # Save the sample data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            sample_ohlc_data.to_csv(tmp_path, index=False)
        
        try:
            # 1. Load data
            loader = DataLoader()
            data = loader.load_data(tmp_path)
            
            # Verify data is loaded correctly
            assert not data.empty
            assert data.shape == sample_ohlc_data.shape
            assert "open" in data.columns
            
            # 2. Analyze data
            analyzer = DataAnalyzer()
            analysis_results = analyzer.analyze(tmp_path)
            
            # Verify analysis results
            assert "summary_stats" in analysis_results
            assert "time_series" in analysis_results
            assert "correlation" in analysis_results
            
            # 3. Train prediction model
            predictor = Predictor()
            
            # 3.1 Linear model
            linear_train_results = predictor.train(
                data, 
                target_column="close",
                features=["open", "high", "low", "volume"],
                method="linear"
            )
            
            # Verify training results
            assert linear_train_results["method"] == "linear"
            assert linear_train_results["target"] == "close"
            assert "metrics" in linear_train_results
            assert "coefficients" in linear_train_results
            
            # 3.2 ARIMA model
            arima_train_results = predictor.train(
                data,
                target_column="close",
                method="arima"
            )
            
            # Verify training results
            assert arima_train_results["method"] == "arima"
            assert arima_train_results["target"] == "close"
            assert "aic" in arima_train_results
            
            # 4. Make predictions
            # 4.1 Linear predictions
            linear_predictions = predictor.predict(
                data,
                target_column="close",
                horizon=3,
                method="linear"
            )
            
            # Verify linear predictions
            assert linear_predictions["method"] == "linear"
            assert linear_predictions["target"] == "close"
            assert "current_predictions" in linear_predictions
            assert "future_predictions" in linear_predictions
            assert len(linear_predictions["future_predictions"]) == 3
            
            # 4.2 ARIMA predictions
            arima_predictions = predictor.predict(
                data,
                target_column="close",
                horizon=3,
                method="arima"
            )
            
            # Verify ARIMA predictions
            assert arima_predictions["method"] == "arima"
            assert arima_predictions["target"] == "close"
            assert "forecast" in arima_predictions
            assert len(arima_predictions["forecast"]) == 3
            
        finally:
            # Clean up the temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_data_loader_analyzer_integration(self):
        """Test integration between DataLoader and DataAnalyzer."""
        # Create sample data
        data_content = b"""date,value
2022-01-01,100
2022-01-02,102
2022-01-03,101
2022-01-04,103
2022-01-05,105"""
        
        # 1. Load the data
        loader = DataLoader()
        data = loader.load_from_content(data_content)
        
        # Verify data loaded correctly
        assert len(data) == 5
        assert "date" in data.columns
        assert "value" in data.columns
        
        # 2. Analyze the data
        analyzer = DataAnalyzer()
        results = analyzer._run_analysis(data, {})
        
        # Verify analysis results
        assert "summary_stats" in results
        assert "time_series" in results
        
        # Check time_series analysis
        assert results["time_series"]["has_time_data"] is True
        
        # 3. Run advanced analysis
        advanced_results = analyzer._run_analysis(data, {"advanced": True})
        
        # Verify advanced results
        assert "advanced" in advanced_results
        assert "volatility" in advanced_results["advanced"]
        assert "trends" in advanced_results["advanced"]
        assert "value" in advanced_results["advanced"]["volatility"]