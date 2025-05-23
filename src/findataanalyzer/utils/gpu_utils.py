"""ماژول مدیریت GPU برای تنظیم و کنترل پردازنده‌های گرافیکی.

این ماژول قابلیت‌های مدیریت GPU، تنظیم حافظه و بهینه‌سازی عملکرد را فراهم می‌کند.
"""

import torch
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class GPUManager:
    """مدیریت GPU و تنظیمات مربوط به آن."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر GPU.
        
        Args:
            config: تنظیمات GPU از فایل پیکربندی
        """
        self.config = config
        self.device = None
        self.initialize()
    
    def initialize(self) -> None:
        """راه‌اندازی و تنظیم اولیه GPU."""
        try:
            if not self.config['gpu']['enabled']:
                self.device = torch.device('cpu')
                logger.info("GPU غیرفعال است. استفاده از CPU")
                return
            
            if torch.cuda.is_available():
                self.device = torch.device(self.config['gpu']['device'])
                
                # تنظیم حافظه GPU
                if self.config['gpu']['memory_fraction'] < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.config['gpu']['memory_fraction']
                    )
                
                # تنظیم رشد حافظه
                if self.config['gpu']['allow_growth']:
                    torch.cuda.empty_cache()
                
                logger.info(f"GPU فعال شد: {torch.cuda.get_device_name(0)}")
                logger.info(f"حافظه کل GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.device = torch.device('cpu')
                logger.warning("GPU در دسترس نیست. استفاده از CPU")
        
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی GPU: {str(e)}")
            self.device = torch.device('cpu')
    
    def get_device(self) -> torch.device:
        """
        دریافت دستگاه محاسباتی فعلی.
        
        Returns:
            دستگاه محاسباتی (GPU یا CPU)
        """
        return self.device
    
    def clear_memory(self) -> None:
        """پاکسازی حافظه GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("حافظه GPU پاکسازی شد")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        دریافت اطلاعات حافظه GPU.
        
        Returns:
            دیکشنری شامل اطلاعات حافظه
        """
        if not torch.cuda.is_available():
            return {
                'total': 0,
                'allocated': 0,
                'cached': 0
            }
        
        return {
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'cached': torch.cuda.memory_reserved(0) / 1024**3
        }
    
    def set_memory_fraction(self, fraction: float) -> None:
        """
        تنظیم سهم حافظه GPU.
        
        Args:
            fraction: سهم حافظه (بین 0 و 1)
        """
        if torch.cuda.is_available() and 0 < fraction <= 1:
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"سهم حافظه GPU به {fraction:.2f} تنظیم شد") 