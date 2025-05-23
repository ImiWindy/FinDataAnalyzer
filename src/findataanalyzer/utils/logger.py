"""ماژول مدیریت لاگ‌ها برای ثبت و مدیریت رویدادهای سیستم.

این ماژول قابلیت‌های ثبت لاگ، مدیریت فایل‌های لاگ و تنظیم سطح لاگ را فراهم می‌کند.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class LogManager:
    """مدیریت لاگ‌ها و تنظیمات مربوط به آن."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر لاگ.
        
        Args:
            config: تنظیمات لاگ از فایل پیکربندی
        """
        self.config = config
        self.logger = None
        self.initialize()
    
    def initialize(self) -> None:
        """راه‌اندازی و تنظیم اولیه سیستم لاگ."""
        try:
            # ایجاد دایرکتوری لاگ
            log_dir = Path(self.config['logging']['file']).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # تنظیم فرمت لاگ
            log_format = logging.Formatter(self.config['logging']['format'])
            
            # تنظیم سطح لاگ
            log_level = getattr(logging, self.config['logging']['level'].upper())
            
            # ایجاد هندلر فایل با چرخش خودکار
            file_handler = RotatingFileHandler(
                self.config['logging']['file'],
                maxBytes=self.config['logging']['max_bytes'],
                backupCount=self.config['logging']['backup_count']
            )
            file_handler.setFormatter(log_format)
            
            # ایجاد هندلر کنسول
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            
            # تنظیم لاگر اصلی
            self.logger = logging.getLogger('findataanalyzer')
            self.logger.setLevel(log_level)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("سیستم لاگ با موفقیت راه‌اندازی شد")
        
        except Exception as e:
            print(f"خطا در راه‌اندازی سیستم لاگ: {str(e)}")
            sys.exit(1)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        دریافت یک لاگر.
        
        Args:
            name: نام لاگر (اختیاری)
            
        Returns:
            شیء لاگر
        """
        if name:
            return logging.getLogger(f"findataanalyzer.{name}")
        return self.logger
    
    def set_level(self, level: str) -> None:
        """
        تنظیم سطح لاگ.
        
        Args:
            level: سطح لاگ (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper())
        self.logger.setLevel(log_level)
        self.logger.info(f"سطح لاگ به {level} تغییر کرد")
    
    def add_file_handler(self, file_path: str, level: Optional[str] = None) -> None:
        """
        افزودن یک هندلر فایل جدید.
        
        Args:
            file_path: مسیر فایل لاگ
            level: سطح لاگ (اختیاری)
        """
        try:
            log_format = logging.Formatter(self.config['logging']['format'])
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=self.config['logging']['max_bytes'],
                backupCount=self.config['logging']['backup_count']
            )
            file_handler.setFormatter(log_format)
            
            if level:
                file_handler.setLevel(getattr(logging, level.upper()))
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"هندلر فایل جدید به {file_path} اضافه شد")
        
        except Exception as e:
            self.logger.error(f"خطا در افزودن هندلر فایل: {str(e)}")
    
    def remove_file_handler(self, file_path: str) -> None:
        """
        حذف یک هندلر فایل.
        
        Args:
            file_path: مسیر فایل لاگ
        """
        for handler in self.logger.handlers[:]:
            if isinstance(handler, RotatingFileHandler) and handler.baseFilename == file_path:
                self.logger.removeHandler(handler)
                self.logger.info(f"هندلر فایل {file_path} حذف شد")
                break 