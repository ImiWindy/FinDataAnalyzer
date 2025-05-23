"""Pytest configuration for FinDataAnalyzer tests."""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add the src directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ["FINDATAANALYZER_CONFIG"] = str(Path(__file__).parent / "test_data" / "test_config.yaml")
os.environ["FINDATAANALYZER_DB_URL"] = "sqlite:///:memory:"
os.environ["FINDATAANALYZER_LOG_LEVEL"] = "DEBUG"


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=10),
        "price": [100, 102, 101, 103, 105, 104, 107, 108, 106, 110],
        "volume": [1000, 1200, 900, 1100, 1300, 950, 1400, 1500, 1100, 1600],
    })


@pytest.fixture
def sample_ohlc_data():
    """Create a sample OHLC DataFrame for testing."""
    return pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=10),
        "open": [100, 102, 101, 103, 105, 104, 107, 108, 106, 110],
        "high": [102, 104, 103, 105, 107, 106, 109, 110, 108, 112],
        "low": [99, 101, 100, 102, 104, 103, 106, 107, 105, 109],
        "close": [102, 101, 103, 105, 104, 107, 108, 106, 110, 111],
        "volume": [1000, 1200, 900, 1100, 1300, 950, 1400, 1500, 1100, 1600],
    }) 