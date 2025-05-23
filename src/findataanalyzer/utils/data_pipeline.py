"""ماژول خط لوله داده.

این ماژول مسئول جمع‌آوری، پردازش و آماده‌سازی داده‌ها برای آموزش مدل و تحلیل است.
"""

import os
import time
import logging
import schedule
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from findataanalyzer.utils.data_manager import DataManager
from findataanalyzer.image_analysis.collectors.chart_collector import TradingViewCollector, BinanceCollector, MetaTraderCollector
from findataanalyzer.image_analysis.processors.image_processor import StandardProcessor, ChartAugmentor


class DataPipeline:
    """خط لوله داده برای جمع‌آوری و پردازش خودکار داده‌ها."""
    
    def __init__(self, config: Dict[str, Any], data_manager: Optional[DataManager] = None):
        """
        مقداردهی اولیه خط لوله داده.
        
        Args:
            config: تنظیمات خط لوله داده
            data_manager: مدیر داده (اختیاری)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager or DataManager(config)
        
        # تنظیم مسیرهای داده
        self.raw_data_dir = Path(config['raw_data_dir'])
        self.processed_data_dir = Path(config['processed_data_dir'])
        self.cache_dir = Path(config['cache_dir'])
        
        # ایجاد جمع‌کننده‌های داده
        self.trading_view_collector = TradingViewCollector(
            output_dir=str(self.raw_data_dir / "charts/tradingview")
        )
        self.binance_collector = BinanceCollector(
            output_dir=str(self.raw_data_dir / "charts/binance")
        )
        self.metatrader_collector = MetaTraderCollector(
            output_dir=str(self.raw_data_dir / "charts/metatrader")
        )
        
        # ایجاد پردازشگرهای تصویر
        self.image_processor = StandardProcessor(
            input_dir=str(self.raw_data_dir / "charts"),
            output_dir=str(self.processed_data_dir / "charts/standard")
        )
        self.image_augmentor = ChartAugmentor(
            input_dir=str(self.processed_data_dir / "charts/standard"),
            output_dir=str(self.processed_data_dir / "charts/augmented")
        )
        
        # تنظیم زمانبندی
        self.scheduler_thread = None
        self.is_running = False
        
        self.logger.info("خط لوله داده با موفقیت راه‌اندازی شد")
    
    def collect_market_data(self, symbols: List[str], 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           source: str = "binance") -> pd.DataFrame:
        """
        جمع‌آوری داده‌های بازار.
        
        Args:
            symbols: لیست نمادها
            start_date: تاریخ شروع (اختیاری)
            end_date: تاریخ پایان (اختیاری)
            source: منبع داده (binance, yahoo, etc.)
            
        Returns:
            داده‌های بازار
        """
        self.logger.info(f"جمع‌آوری داده‌های بازار برای {len(symbols)} نماد از منبع {source}")
        
        # تنظیم تاریخ‌ها
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # برای هر نماد داده‌ها را جمع‌آوری کن
        all_data = []
        for symbol in symbols:
            try:
                # بر اساس منبع داده، از API مناسب استفاده کن
                if source.lower() == "binance":
                    # این بخش نیاز به پیاده‌سازی API بایننس دارد
                    # در اینجا فقط یک پلیس‌هولدر قرار می‌دهیم
                    self.logger.info(f"جمع‌آوری داده‌های {symbol} از بایننس")
                    # data = self._collect_from_binance(symbol, start_date, end_date)
                    data = pd.DataFrame()  # پلیس‌هولدر
                
                elif source.lower() == "yahoo":
                    # این بخش نیاز به پیاده‌سازی API یاهو فایننس دارد
                    self.logger.info(f"جمع‌آوری داده‌های {symbol} از یاهو فایننس")
                    # data = self._collect_from_yahoo(symbol, start_date, end_date)
                    data = pd.DataFrame()  # پلیس‌هولدر
                
                else:
                    raise ValueError(f"منبع داده نامعتبر: {source}")
                
                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
                
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری داده‌های {symbol}: {e}")
        
        # ترکیب داده‌های تمام نمادها
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # ذخیره داده‌های خام
            for symbol in symbols:
                symbol_data = combined_data[combined_data['symbol'] == symbol]
                if not symbol_data.empty:
                    self.data_manager.save_raw_data(symbol, symbol_data)
            
            return combined_data
        
        return pd.DataFrame()
    
    def collect_chart_images(self, symbols: List[str], 
                            timeframes: List[str] = ["1d", "4h", "1h"],
                            sources: List[str] = ["tradingview"]) -> List[str]:
        """
        جمع‌آوری تصاویر نمودار.
        
        Args:
            symbols: لیست نمادها
            timeframes: لیست بازه‌های زمانی
            sources: لیست منابع داده
            
        Returns:
            لیست مسیرهای تصاویر جمع‌آوری شده
        """
        self.logger.info(f"جمع‌آوری تصاویر نمودار برای {len(symbols)} نماد و {len(timeframes)} بازه زمانی")
        
        collected_images = []
        
        for source in sources:
            try:
                if source.lower() == "tradingview":
                    images = self.trading_view_collector.collect(symbols=symbols, timeframes=timeframes)
                    collected_images.extend(images)
                
                elif source.lower() == "binance":
                    images = self.binance_collector.collect(symbols=symbols, timeframes=timeframes)
                    collected_images.extend(images)
                
                elif source.lower() == "metatrader":
                    images = self.metatrader_collector.collect(symbols=symbols, timeframes=timeframes)
                    collected_images.extend(images)
                
                else:
                    self.logger.warning(f"منبع تصویر نامعتبر: {source}")
            
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری تصاویر از {source}: {e}")
        
        return collected_images
    
    def process_chart_images(self, image_paths: List[str], 
                           grayscale: bool = True,
                           normalize: bool = True,
                           augment: bool = True,
                           num_augmentations: int = 3) -> List[str]:
        """
        پردازش تصاویر نمودار.
        
        Args:
            image_paths: لیست مسیرهای تصاویر
            grayscale: تبدیل به سطح خاکستری
            normalize: نرمال‌سازی پیکسل‌ها
            augment: افزایش داده
            num_augmentations: تعداد افزایش داده
            
        Returns:
            لیست مسیرهای تصاویر پردازش شده
        """
        self.logger.info(f"پردازش {len(image_paths)} تصویر نمودار")
        
        processed_images = []
        
        # پردازش استاندارد
        for image_path in image_paths:
            try:
                processed_path = self.image_processor.process(
                    image_path, grayscale=grayscale, normalize=normalize
                )
                processed_images.append(processed_path)
                
                # افزایش داده
                if augment:
                    augmented_paths = self.image_augmentor.augment_multiple(
                        processed_path, num_augmentations=num_augmentations
                    )
                    processed_images.extend(augmented_paths)
            
            except Exception as e:
                self.logger.error(f"خطا در پردازش تصویر {image_path}: {e}")
        
        return processed_images
    
    def process_market_data(self, symbols: List[str], 
                          add_technical_indicators: bool = True) -> Dict[str, pd.DataFrame]:
        """
        پردازش داده‌های بازار.
        
        Args:
            symbols: لیست نمادها
            add_technical_indicators: افزودن شاخص‌های تکنیکال
            
        Returns:
            دیکشنری داده‌های پردازش شده
        """
        self.logger.info(f"پردازش داده‌های بازار برای {len(symbols)} نماد")
        
        processed_data = {}
        
        for symbol in symbols:
            try:
                # بارگذاری داده‌های خام
                raw_data = self.data_manager.load_raw_data(symbol)
                
                if raw_data is None or raw_data.empty:
                    self.logger.warning(f"داده‌ای برای نماد {symbol} یافت نشد")
                    continue
                
                # افزودن شاخص‌های تکنیکال
                if add_technical_indicators:
                    data = self._add_technical_indicators(raw_data)
                else:
                    data = raw_data.copy()
                
                # ذخیره داده‌های پردازش شده
                self.data_manager.save_processed_data(symbol, data)
                
                processed_data[symbol] = data
            
            except Exception as e:
                self.logger.error(f"خطا در پردازش داده‌های {symbol}: {e}")
        
        return processed_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        افزودن شاخص‌های تکنیکال به داده‌ها.
        
        Args:
            data: داده‌های خام
            
        Returns:
            داده‌ها با شاخص‌های تکنیکال
        """
        self.logger.info("افزودن شاخص‌های تکنیکال")
        
        # اطمینان از وجود ستون‌های مورد نیاز
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"ستون‌های مورد نیاز برای محاسبه شاخص‌های تکنیکال یافت نشد: {missing_columns}")
            return data
        
        result = data.copy()
        
        try:
            # در اینجا از یک کتابخانه مانند ta-lib یا pandas_ta استفاده می‌شود
            # برای سادگی، چند نمونه شاخص را با روش‌های پایه محاسبه می‌کنیم
            
            # میانگین متحرک ساده
            for window in [7, 14, 21, 50, 200]:
                result[f'sma_{window}'] = result['close'].rolling(window=window).mean()
            
            # میانگین متحرک نمایی
            for window in [7, 14, 21, 50, 200]:
                result[f'ema_{window}'] = result['close'].ewm(span=window, adjust=False).mean()
            
            # قدرت نسبی (RSI)
            def calculate_rsi(series, window=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            result['rsi_14'] = calculate_rsi(result['close'], 14)
            
            # میانگین محدوده واقعی (ATR)
            def calculate_atr(df, window=14):
                high = df['high']
                low = df['low']
                close = df['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.rolling(window=window).mean()
                return atr
            
            result['atr_14'] = calculate_atr(result, 14)
            
            # باندهای بولینگر
            def calculate_bollinger_bands(series, window=20, num_std=2):
                middle_band = series.rolling(window=window).mean()
                std_dev = series.rolling(window=window).std()
                upper_band = middle_band + (num_std * std_dev)
                lower_band = middle_band - (num_std * std_dev)
                return middle_band, upper_band, lower_band
            
            result['bb_middle_20'], result['bb_upper_20'], result['bb_lower_20'] = calculate_bollinger_bands(result['close'], 20, 2)
            
        except Exception as e:
            self.logger.error(f"خطا در محاسبه شاخص‌های تکنیکال: {e}")
        
        return result
    
    def schedule_data_collection(self, symbols: List[str], 
                               interval: str = "4h",
                               schedule_time: str = "00:00") -> None:
        """
        زمانبندی جمع‌آوری خودکار داده‌ها.
        
        Args:
            symbols: لیست نمادها
            interval: بازه زمانی جمع‌آوری ("1h", "4h", "1d")
            schedule_time: زمان شروع جمع‌آوری (در فرمت 24 ساعته)
        """
        self.logger.info(f"زمانبندی جمع‌آوری داده‌ها برای {len(symbols)} نماد با بازه {interval}")
        
        # ایجاد تابع جمع‌آوری داده
        def collect_job():
            try:
                self.logger.info("شروع جمع‌آوری خودکار داده‌ها")
                
                # جمع‌آوری داده‌های بازار
                market_data = self.collect_market_data(symbols)
                
                # جمع‌آوری تصاویر نمودار
                timeframes = ["1d", "4h", "1h"]  # می‌تواند بر اساس نیاز تنظیم شود
                chart_images = self.collect_chart_images(symbols, timeframes)
                
                # پردازش داده‌ها
                if not market_data.empty:
                    self.process_market_data(symbols)
                
                if chart_images:
                    self.process_chart_images(chart_images)
                
                self.logger.info("جمع‌آوری خودکار داده‌ها با موفقیت انجام شد")
            
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری خودکار داده‌ها: {e}")
        
        # تنظیم زمانبندی بر اساس بازه زمانی
        if interval.lower() == "1h":
            schedule.every(1).hour.do(collect_job)
        elif interval.lower() == "4h":
            schedule.every(4).hours.do(collect_job)
        elif interval.lower() == "1d":
            schedule.every().day.at(schedule_time).do(collect_job)
        else:
            raise ValueError(f"بازه زمانی نامعتبر: {interval}")
        
        # اجرای کار در زمان مشخص شده
        def run_scheduler():
            self.is_running = True
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # بررسی هر دقیقه
        
        # شروع ترد زمانبندی
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def stop_scheduler(self) -> None:
        """توقف زمانبندی جمع‌آوری داده‌ها."""
        self.logger.info("توقف زمانبندی جمع‌آوری داده‌ها")
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # پاکسازی همه کارهای زمانبندی شده
        schedule.clear()
    
    def prepare_training_data(self, symbols: List[str],
                            target_column: str,
                            feature_columns: Optional[List[str]] = None,
                            sequence_length: int = 20,
                            train_split: float = 0.7,
                            val_split: float = 0.15) -> Dict[str, Any]:
        """
        آماده‌سازی داده‌ها برای آموزش مدل.
        
        Args:
            symbols: لیست نمادها
            target_column: ستون هدف
            feature_columns: لیست ستون‌های ویژگی (اختیاری)
            sequence_length: طول دنباله
            train_split: نسبت داده‌های آموزش
            val_split: نسبت داده‌های اعتبارسنجی
            
        Returns:
            دیکشنری داده‌های آموزش
        """
        self.logger.info(f"آماده‌سازی داده‌های آموزش برای {len(symbols)} نماد")
        
        training_data = {}
        
        for symbol in symbols:
            try:
                # بررسی وجود داده‌های کش شده
                cache_key = f"training_data_{symbol}_{target_column}_{sequence_length}"
                cached_data = self.data_manager.load_from_cache(cache_key)
                
                if cached_data is not None:
                    self.logger.info(f"بارگذاری داده‌های آموزش از کش برای {symbol}")
                    training_data[symbol] = cached_data
                    continue
                
                # آماده‌سازی داده‌های آموزش
                result = self.data_manager.prepare_training_data(
                    symbol=symbol,
                    sequence_length=sequence_length,
                    target_column=target_column,
                    feature_columns=feature_columns or [],
                    train_split=train_split,
                    val_split=val_split
                )
                
                # ذخیره در کش
                self.data_manager.save_to_cache(cache_key, result)
                
                training_data[symbol] = result
            
            except Exception as e:
                self.logger.error(f"خطا در آماده‌سازی داده‌های آموزش برای {symbol}: {e}")
        
        return training_data 