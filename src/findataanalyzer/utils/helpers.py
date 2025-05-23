"""Helper functions for the application."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import hashlib
import json
from datetime import datetime


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary with logging settings
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file")
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        filename=log_file
    )


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate DataFrame for analysis.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
        
    Raises:
        ValueError: If data is invalid
    """
    # Check if data is empty
    if data is None or data.empty:
        raise ValueError("Data is empty")
    
    # Check if data has at least one numerical column
    numerical_cols = data.select_dtypes(include=['number']).columns
    if len(numerical_cols) == 0:
        raise ValueError("Data must have at least one numerical column")
    
    return True


def generate_data_hash(data: pd.DataFrame) -> str:
    """
    Generate a hash of the DataFrame for caching purposes.
    
    Args:
        data: DataFrame to hash
        
    Returns:
        Hash string
    """
    # Convert DataFrame to JSON and hash it
    data_json = data.to_json(orient="records")
    return hashlib.md5(data_json.encode()).hexdigest()


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted timestamp string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def handle_missing_values(data: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        data: DataFrame with missing values
        strategy: Strategy for handling missing values (mean, median, mode, drop)
        
    Returns:
        DataFrame with missing values handled
    """
    if data is None or data.empty:
        return data
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    if strategy == "drop":
        # Drop rows with missing values
        result = result.dropna()
    else:
        # Fill missing values
        for column in result.columns:
            if result[column].dtype.kind in 'ifc':  # Integer, float, complex
                if strategy == "mean":
                    result[column].fillna(result[column].mean(), inplace=True)
                elif strategy == "median":
                    result[column].fillna(result[column].median(), inplace=True)
                elif strategy == "mode":
                    result[column].fillna(result[column].mode()[0], inplace=True)
                else:
                    result[column].fillna(0, inplace=True)
            else:
                # For non-numeric columns, fill with most common value
                if not result[column].empty:
                    result[column].fillna(result[column].mode()[0], inplace=True)
    
    return result


def normalize_data(data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize numerical columns in DataFrame.
    
    Args:
        data: DataFrame to normalize
        columns: List of columns to normalize (None for all numerical columns)
        
    Returns:
        DataFrame with normalized columns
    """
    if data is None or data.empty:
        return data
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Get numerical columns if not specified
    if columns is None:
        columns = result.select_dtypes(include=['number']).columns.tolist()
    
    # Normalize each column
    for column in columns:
        if column in result.columns and result[column].dtype.kind in 'ifc':
            min_val = result[column].min()
            max_val = result[column].max()
            
            if max_val > min_val:
                result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result


def detect_outliers(data: pd.DataFrame, column: str, method: str = "zscore", threshold: float = 3.0) -> List[int]:
    """
    Detect outliers in a DataFrame column.
    
    Args:
        data: DataFrame to analyze
        column: Column to analyze
        method: Method for outlier detection (zscore, iqr)
        threshold: Threshold for outlier detection
        
    Returns:
        List of indices of outliers
    """
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        return []
    
    values = data[column].values
    
    if method == "zscore":
        # Z-score method
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values - mean) / std)
        return list(np.where(z_scores > threshold)[0])
    
    elif method == "iqr":
        # IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return list(np.where((values < lower_bound) | (values > upper_bound))[0])
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")


def serialize_results(results: Dict[str, Any]) -> str:
    """
    Serialize results to JSON with special handling for NumPy types.
    
    Args:
        results: Dictionary with results
        
    Returns:
        JSON string
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    return json.dumps(results, cls=NumpyEncoder) 