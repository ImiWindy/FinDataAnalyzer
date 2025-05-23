"""Data loader module for financial data."""

import pandas as pd
import io
import os
from typing import Dict, Any, Optional, Union
import requests

from findataanalyzer.utils.config import get_config


class DataLoader:
    """Data loader class for financial data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.config = get_config()
    
    def load_data(self, source: str) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: Path or URL to the data source
            
        Returns:
            Pandas DataFrame with loaded data
        """
        # Check if source is a URL
        if source.startswith(('http://', 'https://')):
            return self._load_from_url(source)
            
        # Check if source is a local file
        if os.path.exists(source):
            return self._load_from_file(source)
            
        # If source doesn't match any known pattern, raise an error
        raise ValueError(f"Unknown data source: {source}")
    
    def load_from_content(self, content: bytes) -> pd.DataFrame:
        """
        Load data from raw content.
        
        Args:
            content: Raw data content as bytes
            
        Returns:
            Pandas DataFrame with loaded data
        """
        try:
            # Try different formats
            try:
                return pd.read_csv(io.BytesIO(content))
            except:
                try:
                    return pd.read_excel(io.BytesIO(content))
                except:
                    try:
                        return pd.read_json(io.BytesIO(content))
                    except:
                        raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Failed to load data from content: {str(e)}")
    
    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from a local file."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            raise ValueError(f"Failed to load data from file {file_path}: {str(e)}")
    
    def _load_from_url(self, url: str) -> pd.DataFrame:
        """Load data from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            content = response.content
            
            # Try to infer the format from the URL or content
            if url.endswith('.csv'):
                return pd.read_csv(io.BytesIO(content))
            elif url.endswith(('.xls', '.xlsx')):
                return pd.read_excel(io.BytesIO(content))
            elif url.endswith('.json'):
                return pd.read_json(io.BytesIO(content))
            else:
                # Try to infer format from content
                return self.load_from_content(content)
                
        except Exception as e:
            raise ValueError(f"Failed to load data from URL {url}: {str(e)}") 