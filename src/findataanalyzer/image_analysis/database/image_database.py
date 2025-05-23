"""ماژول پایگاه داده تصاویر برای ذخیره‌سازی و بازیابی تصاویر نمودارهای مالی.

این ماژول با پشتیبانی از پردازش GPU، عملیات پایگاه داده تصاویر نمودار را با سرعت بالا انجام می‌دهد.
"""

import os
import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import sqlite3
from datetime import datetime
import uuid
import hashlib
import tensorflow as tf  # استفاده از TensorFlow به جای PyOpenCL
from PIL import Image
import pickle


# تنظیمات پایه برای استفاده از GPU
def setup_gpu_context():
    """
    راه‌اندازی زمینه پردازش GPU با استفاده از TensorFlow.
    
    این تابع تلاش می‌کند تا یک زمینه GPU برای پردازش با TensorFlow ایجاد کند.
    اگر GPU در دسترس نباشد، به CPU برمی‌گردد.
    
    Returns:
        bool: آیا GPU موجود است
    """
    # سعی در دستیابی به GPU با استفاده از TensorFlow
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # اجازه رشد حافظه
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logging.info(f"GPU برای پردازش پیدا شد: {len(gpus)} دستگاه")
            return True
        else:
            # اگر GPU موجود نیست، از CPU استفاده کن
            logging.warning("GPU یافت نشد، استفاده از CPU")
            return False
    except Exception as e:
        logging.error(f"خطا در راه‌اندازی زمینه GPU: {e}")
        return False


class GPUImageProcessor:
    """کلاس پردازش تصویر با استفاده از GPU."""
    
    def __init__(self):
        """راه‌اندازی پردازشگر تصویر GPU."""
        self.gpu_available = setup_gpu_context()
        self.logger = logging.getLogger(__name__)
        
        if self.gpu_available:
            logging.info("پردازشگر تصویر GPU با موفقیت راه‌اندازی شد")
        else:
            logging.warning("GPU در دسترس نیست، استفاده از پردازش CPU")
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        تغییر اندازه تصویر با استفاده از GPU.
        
        Args:
            image: تصویر ورودی به شکل آرایه numpy
            target_size: اندازه مقصد به شکل (عرض، ارتفاع)
            
        Returns:
            تصویر تغییر اندازه داده شده
        """
        if not self.gpu_available:
            return cv2.resize(image, target_size)
        
        try:
            # تبدیل به تنسور TensorFlow
            image_tensor = tf.convert_to_tensor(image)
            # استفاده از عملیات تغییر اندازه TensorFlow
            resized_tensor = tf.image.resize(image_tensor, target_size)
            # تبدیل به آرایه numpy
            result = resized_tensor.numpy().astype(np.uint8)
            return result
        except Exception as e:
            self.logger.error(f"خطا در تغییر اندازه تصویر با GPU: {e}")
            return cv2.resize(image, target_size)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        تبدیل تصویر به سطح خاکستری با استفاده از GPU.
        
        Args:
            image: تصویر ورودی RGB
            
        Returns:
            تصویر سطح خاکستری
        """
        if not self.gpu_available or len(image.shape) != 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        try:
            # تبدیل به تنسور TensorFlow
            image_tensor = tf.convert_to_tensor(image)
            # استفاده از عملیات تبدیل به سطح خاکستری TensorFlow
            # TensorFlow از فرمول rgb_to_grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B استفاده می‌کند
            grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
            # تبدیل به آرایه numpy
            result = tf.squeeze(grayscale_tensor).numpy().astype(np.uint8)
            return result
        except Exception as e:
            self.logger.error(f"خطا در تبدیل تصویر به سطح خاکستری با GPU: {e}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        نرمال‌سازی تصویر با استفاده از TensorFlow.
        
        Args:
            image: تصویر ورودی
            
        Returns:
            تصویر نرمال‌سازی شده
        """
        if not self.gpu_available:
            # نرمال‌سازی به محدوده [0, 1]
            return image.astype(np.float32) / 255.0
        
        try:
            # تبدیل به تنسور TensorFlow
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            # نرمال‌سازی به محدوده [0, 1]
            normalized_tensor = image_tensor / 255.0
            # تبدیل به آرایه numpy
            result = normalized_tensor.numpy()
            return result
        except Exception as e:
            self.logger.error(f"خطا در نرمال‌سازی تصویر با GPU: {e}")
            return image.astype(np.float32) / 255.0 