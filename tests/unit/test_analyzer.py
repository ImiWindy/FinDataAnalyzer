"""Tests for the analyzer module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from findataanalyzer.core.analyzer import DataAnalyzer


class TestDataAnalyzer:
    """Tests for the DataAnalyzer class."""
    
    def test_init(self):
        """Test initialization of the DataAnalyzer."""
        analyzer = DataAnalyzer()
        assert hasattr(analyzer, "data_loader")
    
    @patch("findataanalyzer.core.analyzer.DataLoader")
    @patch("findataanalyzer.core.analyzer.validate_data")
    def test_analyze(self, mock_validate, mock_loader_class):
        """Test the analyze method."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        mock_df = pd.DataFrame({
            "date": pd.date_range(start="2022-01-01", periods=3),
            "price": [100, 102, 101],
            "volume": [1000, 1200, 900]
        })
        mock_loader.load_data.return_value = mock_df
        
        # Create analyzer and call analyze
        analyzer = DataAnalyzer()
        results = analyzer.analyze("dummy/path.csv")
        
        # Verify calls
        mock_loader.load_data.assert_called_once_with("dummy/path.csv")
        mock_validate.assert_called_once_with(mock_df)
        
        # Verify results
        assert "summary_stats" in results
        assert "time_series" in results
    
    def test_analyze_file(self):
        """Test the analyze_file method."""
        with patch.object(DataAnalyzer, "analyze") as mock_analyze:
            analyzer = DataAnalyzer()
            analyzer.analyze_file("test.csv")
            mock_analyze.assert_called_once_with("test.csv")
    
    def test_analyze_raw_data(self):
        """Test the analyze_raw_data method."""
        with patch.object(DataAnalyzer, "_run_analysis") as mock_run_analysis:
            mock_run_analysis.return_value = {"test": "result"}
            
            # Setup mock for data_loader
            mock_loader = MagicMock()
            mock_df = pd.DataFrame({"a": [1, 2, 3]})
            mock_loader.load_from_content.return_value = mock_df
            
            analyzer = DataAnalyzer()
            analyzer.data_loader = mock_loader
            
            # Call the method
            result = analyzer.analyze_raw_data(b"test data")
            
            # Verify calls
            mock_loader.load_from_content.assert_called_once_with(b"test data")
            mock_run_analysis.assert_called_once_with(mock_df, {})
            
            # Verify result
            assert result == {"test": "result"}
    
    def test_run_analysis(self, sample_data):
        """Test the _run_analysis method."""
        analyzer = DataAnalyzer()
        results = analyzer._run_analysis(sample_data, {})
        
        # Check results
        assert "summary_stats" in results
        assert "time_series" in results
        assert "correlation" in results
        
        # With advanced parameter
        results = analyzer._run_analysis(sample_data, {"advanced": True})
        assert "advanced" in results
        assert "volatility" in results["advanced"]
        assert "trends" in results["advanced"]
    
    def test_analyze_time_series(self, sample_data):
        """Test the _analyze_time_series method."""
        analyzer = DataAnalyzer()
        results = analyzer._analyze_time_series(sample_data, "date")
        
        assert results["has_time_data"] is True
        assert "start" in results["time_range"]
        assert "end" in results["time_range"]
    
    def test_calculate_volatility(self, sample_data):
        """Test the _calculate_volatility method."""
        analyzer = DataAnalyzer()
        volatility = analyzer._calculate_volatility(sample_data)
        
        assert "price" in volatility
        assert "volume" in volatility
        assert volatility["price"] > 0
    
    def test_identify_trends(self, sample_data):
        """Test the _identify_trends method."""
        analyzer = DataAnalyzer()
        trends = analyzer._identify_trends(sample_data)
        
        assert "price" in trends
        assert "volume" in trends
        assert trends["price"] in ["upward", "downward", "stable"] 