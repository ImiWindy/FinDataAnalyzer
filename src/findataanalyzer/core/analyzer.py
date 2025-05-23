"""Data analyzer module for financial data processing."""

import pandas as pd
from typing import Dict, Any, List, Optional

from findataanalyzer.core.data_loader import DataLoader
from findataanalyzer.utils.helpers import validate_data


class DataAnalyzer:
    """Main data analyzer class for financial data."""
    
    def __init__(self):
        """Initialize the data analyzer."""
        self.data_loader = DataLoader()
    
    def analyze(self, data_source: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze financial data from a specified source.
        
        Args:
            data_source: Path or URL to the data source
            parameters: Optional parameters for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        # Load the data
        data = self.data_loader.load_data(data_source)
        
        # Validate the data
        validate_data(data)
        
        # Perform analysis
        return self._run_analysis(data, parameters or {})
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze data from a local file."""
        return self.analyze(file_path)
    
    def analyze_raw_data(self, data_content: bytes) -> Dict[str, Any]:
        """Analyze raw data content."""
        data = self.data_loader.load_from_content(data_content)
        return self._run_analysis(data, {})
    
    def _run_analysis(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis on the provided data."""
        results = {}
        
        # Basic statistics
        results["summary_stats"] = data.describe().to_dict()
        
        # Time series analysis if time column exists
        if "date" in data.columns or "time" in data.columns:
            time_col = "date" if "date" in data.columns else "time"
            results["time_series"] = self._analyze_time_series(data, time_col)
        
        # Correlation analysis for numerical columns
        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) > 1:
            results["correlation"] = data[numerical_cols].corr().to_dict()
            
        # Add any additional analysis based on parameters
        if "advanced" in parameters and parameters["advanced"]:
            results["advanced"] = self._run_advanced_analysis(data)
            
        return results
    
    def _analyze_time_series(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Analyze time series data."""
        # Basic implementation, to be expanded
        return {
            "has_time_data": True,
            "time_range": {
                "start": data[time_column].min().isoformat() if hasattr(data[time_column].min(), 'isoformat') else str(data[time_column].min()),
                "end": data[time_column].max().isoformat() if hasattr(data[time_column].max(), 'isoformat') else str(data[time_column].max()),
            }
        }
    
    def _run_advanced_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run advanced analysis on the data."""
        # Placeholder for more complex analysis
        return {
            "volatility": self._calculate_volatility(data),
            "trends": self._identify_trends(data),
        }
    
    def _calculate_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics."""
        # Basic implementation
        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        return {col: float(data[col].std()) for col in numerical_cols}
    
    def _identify_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Identify trends in the data."""
        # Placeholder implementation
        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        trends = {}
        
        for col in numerical_cols:
            if len(data) > 1:
                first_val = data[col].iloc[0]
                last_val = data[col].iloc[-1]
                
                if last_val > first_val:
                    trends[col] = "upward"
                elif last_val < first_val:
                    trends[col] = "downward"
                else:
                    trends[col] = "stable"
            else:
                trends[col] = "insufficient_data"
        
        return trends 