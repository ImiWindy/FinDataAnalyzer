"""ماژول مدیریت داده‌ها.

این ماژول مسئول مدیریت داده‌های خام، پردازش شده و کش شده است.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DataManager:
    """مدیریت داده‌ها."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر داده.
        
        Args:
            config: تنظیمات مدیریت داده
        """
        self.config = config
        self.raw_data_dir = Path(config['raw_data_dir'])
        self.processed_data_dir = Path(config['processed_data_dir'])
        self.cache_dir = Path(config['cache_dir'])
        
        # ایجاد دایرکتوری‌های مورد نیاز
        self._create_directories()
    
    def _create_directories(self) -> None:
        """ایجاد دایرکتوری‌های مورد نیاز."""
        for directory in [self.raw_data_dir, self.processed_data_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        ذخیره داده‌های خام.
        
        Args:
            symbol: نماد
            data: داده‌های خام
        """
        file_path = self.raw_data_dir / f"{symbol}.csv"
        data.to_csv(file_path, index=True)
    
    def load_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        بارگذاری داده‌های خام.
        
        Args:
            symbol: نماد
            
        Returns:
            داده‌های خام
        """
        file_path = self.raw_data_dir / f"{symbol}.csv"
        if not file_path.exists():
            return None
        
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def save_processed_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        ذخیره داده‌های پردازش شده.
        
        Args:
            symbol: نماد
            data: داده‌های پردازش شده
        """
        file_path = self.processed_data_dir / f"{symbol}.csv"
        data.to_csv(file_path, index=True)
    
    def load_processed_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        بارگذاری داده‌های پردازش شده.
        
        Args:
            symbol: نماد
            
        Returns:
            داده‌های پردازش شده
        """
        file_path = self.processed_data_dir / f"{symbol}.csv"
        if not file_path.exists():
            return None
        
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def save_to_cache(self, key: str, data: Any) -> None:
        """
        ذخیره داده در کش.
        
        Args:
            key: کلید کش
            data: داده
        """
        file_path = self.cache_dir / f"{key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_cache(self, key: str) -> Optional[Any]:
        """
        بارگذاری داده از کش.
        
        Args:
            key: کلید کش
            
        Returns:
            داده
        """
        file_path = self.cache_dir / f"{key}.pkl"
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self) -> None:
        """پاک کردن کش."""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """
        دریافت محدوده زمانی داده‌ها.
        
        Args:
            symbol: نماد
            
        Returns:
            زمان شروع و پایان
        """
        data = self.load_processed_data(symbol)
        if data is None:
            raise ValueError(f"داده‌ای برای نماد {symbol} یافت نشد")
        
        return data.index.min(), data.index.max()
    
    def get_available_symbols(self) -> List[str]:
        """
        دریافت لیست نمادهای موجود.
        
        Returns:
            لیست نمادها
        """
        return [f.stem for f in self.processed_data_dir.glob("*.csv")]
    
    def prepare_training_data(self, symbol: str, 
                            sequence_length: int,
                            target_column: str,
                            feature_columns: List[str],
                            train_split: float = 0.7,
                            val_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray]:
        """
        آماده‌سازی داده‌های آموزش.
        
        Args:
            symbol: نماد
            sequence_length: طول دنباله
            target_column: ستون هدف
            feature_columns: ستون‌های ویژگی
            train_split: نسبت داده‌های آموزش
            val_split: نسبت داده‌های اعتبارسنجی
            
        Returns:
            داده‌های آموزش، اعتبارسنجی و آزمون
        """
        data = self.load_processed_data(symbol)
        if data is None:
            raise ValueError(f"داده‌ای برای نماد {symbol} یافت نشد")
        
        # آماده‌سازی داده‌ها
        X = data[feature_columns].values
        y = data[target_column].values
        
        # ایجاد دنباله‌ها
        X_sequences = []
        y_sequences = []
        
        for i in range(len(data) - sequence_length):
            X_sequences.append(X[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # تقسیم داده‌ها
        train_size = int(len(X_sequences) * train_split)
        val_size = int(len(X_sequences) * val_split)
        
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        
        X_val = X_sequences[train_size:train_size + val_size]
        y_val = y_sequences[train_size:train_size + val_size]
        
        X_test = X_sequences[train_size + val_size:]
        y_test = y_sequences[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_training_data(self, symbol: str, data: Tuple[np.ndarray, ...]) -> None:
        """
        ذخیره داده‌های آموزش.
        
        Args:
            symbol: نماد
            data: داده‌های آموزش
        """
        cache_key = f"training_data_{symbol}"
        self.save_to_cache(cache_key, data)
    
    def load_training_data(self, symbol: str) -> Optional[Tuple[np.ndarray, ...]]:
        """
        بارگذاری داده‌های آموزش.
        
        Args:
            symbol: نماد
            
        Returns:
            داده‌های آموزش
        """
        cache_key = f"training_data_{symbol}"
        return self.load_from_cache(cache_key)
    
    def get_data_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        دریافت آمار داده‌ها.
        
        Args:
            symbol: نماد
            
        Returns:
            آمار داده‌ها
        """
        data = self.load_processed_data(symbol)
        if data is None:
            raise ValueError(f"داده‌ای برای نماد {symbol} یافت نشد")
        
        return {
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'total_samples': len(data),
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'statistics': data.describe().to_dict()
        } 