"""ماژول جمع‌آوری داده‌های بازار.

این ماژول امکان جمع‌آوری داده‌های بازار از منابع مختلف را فراهم می‌کند.
"""

import os
import pandas as pd
import numpy as np
import logging
import requests
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class MarketDataCollector(ABC):
    """کلاس پایه برای جمع‌آوری داده‌های بازار."""
    
    def __init__(self, output_dir: str = "data/raw/market_data"):
        """مقداردهی اولیه کلاس جمع‌کننده داده.
        
        Args:
            output_dir: مسیر ذخیره‌سازی داده‌ها
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def collect_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """جمع‌آوری داده‌های بازار.
        
        Args:
            symbol: نماد مورد نظر
            start_date: تاریخ شروع
            end_date: تاریخ پایان
            interval: بازه زمانی (روزانه، ساعتی و غیره)
            
        Returns:
            دیتافریم داده‌های جمع‌آوری شده
        """
        pass
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str) -> str:
        """ذخیره داده‌های جمع‌آوری شده.
        
        Args:
            data: دیتافریم داده‌ها
            symbol: نماد
            interval: بازه زمانی
            
        Returns:
            مسیر فایل ذخیره شده
        """
        if data.empty:
            self.logger.warning(f"داده‌ای برای ذخیره‌سازی نماد {symbol} وجود ندارد.")
            return ""
        
        # ایجاد نام فایل با فرمت symbol_interval_startdate_enddate.csv
        start_date = data.index.min().strftime('%Y%m%d')
        end_date = data.index.max().strftime('%Y%m%d')
        filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        
        # ذخیره به CSV
        file_path = self.output_dir / filename
        data.to_csv(file_path)
        
        self.logger.info(f"داده‌های {symbol} در فایل {file_path} ذخیره شد.")
        return str(file_path)


class YahooFinanceCollector(MarketDataCollector):
    """جمع‌کننده داده از یاهو فایننس."""
    
    def __init__(self, output_dir: str = "data/raw/market_data/yahoo"):
        """مقداردهی اولیه کلاس جمع‌کننده یاهو فایننس.
        
        Args:
            output_dir: مسیر ذخیره‌سازی داده‌ها
        """
        super().__init__(output_dir)
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    def collect_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """جمع‌آوری داده‌های بازار از یاهو فایننس.
        
        Args:
            symbol: نماد مورد نظر
            start_date: تاریخ شروع
            end_date: تاریخ پایان
            interval: بازه زمانی (1d, 1h, 5m و غیره)
            
        Returns:
            دیتافریم داده‌های جمع‌آوری شده
        """
        self.logger.info(f"جمع‌آوری داده‌های {symbol} از یاهو فایننس...")
        
        # تبدیل تاریخ به تایم‌استمپ یونیکس
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # ساخت پارامترهای درخواست
        params = {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": interval,
            "includePrePost": "true",
            "events": "div,splits"
        }
        
        try:
            # ارسال درخواست
            response = requests.get(f"{self.base_url}{symbol}", params=params)
            response.raise_for_status()
            
            # پردازش پاسخ
            data = response.json()
            
            # بررسی اعتبار پاسخ
            if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
                self.logger.error(f"خطا در دریافت داده برای {symbol}: پاسخ نامعتبر")
                return pd.DataFrame()
            
            # استخراج داده‌ها
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            
            # ایجاد دیتافریم
            df = pd.DataFrame({
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", [])
            }, index=pd.to_datetime([datetime.fromtimestamp(ts) for ts in timestamps]))
            
            # تنظیم ایندکس
            df.index.name = "date"
            
            self.logger.info(f"جمع‌آوری {len(df)} رکورد برای {symbol} انجام شد.")
            return df
            
        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری داده برای {symbol}: {e}")
            return pd.DataFrame()


class BinanceCollector(MarketDataCollector):
    """جمع‌کننده داده از بایننس."""
    
    def __init__(self, output_dir: str = "data/raw/market_data/binance", api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """مقداردهی اولیه کلاس جمع‌کننده بایننس.
        
        Args:
            output_dir: مسیر ذخیره‌سازی داده‌ها
            api_key: کلید API بایننس (اختیاری)
            api_secret: رمز API بایننس (اختیاری)
        """
        super().__init__(output_dir)
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.api_key = api_key
        self.api_secret = api_secret
        
        # نگاشت بازه‌های زمانی
        self.interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1M"
        }
    
    def collect_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """جمع‌آوری داده‌های بازار از بایننس.
        
        Args:
            symbol: نماد مورد نظر
            start_date: تاریخ شروع
            end_date: تاریخ پایان
            interval: بازه زمانی (1d, 1h, 15m و غیره)
            
        Returns:
            دیتافریم داده‌های جمع‌آوری شده
        """
        self.logger.info(f"جمع‌آوری داده‌های {symbol} از بایننس...")
        
        # تبدیل بازه زمانی به فرمت بایننس
        binance_interval = self.interval_map.get(interval, "1d")
        
        # تبدیل تاریخ به تایم‌استمپ میلی‌ثانیه‌ای
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # آماده‌سازی هدرهای درخواست
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        
        all_klines = []
        current_start = start_timestamp
        
        # به دلیل محدودیت تعداد نتایج بازگشتی، نیاز به چند درخواست داریم
        while current_start < end_timestamp:
            try:
                # پارامترهای درخواست
                params = {
                    "symbol": symbol.upper().replace("-", ""),  # حذف خط تیره برای سازگاری با بایننس
                    "interval": binance_interval,
                    "startTime": current_start,
                    "endTime": end_timestamp,
                    "limit": 1000  # حداکثر تعداد نتایج در هر درخواست
                }
                
                # ارسال درخواست
                response = requests.get(self.base_url, params=params, headers=headers)
                response.raise_for_status()
                
                # تبدیل پاسخ به JSON
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # تنظیم زمان شروع برای درخواست بعدی
                current_start = klines[-1][0] + 1
                
                # رعایت محدودیت نرخ درخواست
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری داده برای {symbol}: {e}")
                break
        
        if not all_klines:
            self.logger.warning(f"هیچ داده‌ای برای {symbol} یافت نشد.")
            return pd.DataFrame()
        
        # تبدیل داده‌های کندل به دیتافریم
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        
        # تبدیل نوع داده‌ها
        numeric_columns = ["open", "high", "low", "close", "volume", 
                         "quote_asset_volume", "taker_buy_base_asset_volume", 
                         "taker_buy_quote_asset_volume"]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # تنظیم ایندکس تاریخ
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        
        # حذف ستون‌های اضافی
        df = df[["open", "high", "low", "close", "volume"]]
        
        self.logger.info(f"جمع‌آوری {len(df)} رکورد برای {symbol} انجام شد.")
        return df


def get_collector(source: str, **kwargs) -> MarketDataCollector:
    """دریافت جمع‌کننده داده مناسب بر اساس منبع.
    
    Args:
        source: منبع داده (yahoo یا binance)
        **kwargs: پارامترهای اضافی برای جمع‌کننده
        
    Returns:
        جمع‌کننده داده مناسب
    """
    if source.lower() == "yahoo":
        return YahooFinanceCollector(**kwargs)
    elif source.lower() == "binance":
        return BinanceCollector(**kwargs)
    else:
        raise ValueError(f"منبع داده نامعتبر: {source}") 