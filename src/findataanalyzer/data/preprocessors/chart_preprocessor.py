"""ماژول پیش‌پردازش تصاویر نمودار.

این ماژول امکانات لازم برای پیش‌پردازش تصاویر نمودارهای مالی را فراهم می‌کند.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


class ChartPreprocessor:
    """کلاس پیش‌پردازش تصاویر نمودار."""
    
    def __init__(self, output_dir: str = "data/processed/charts", target_size: Tuple[int, int] = (224, 224)):
        """مقداردهی اولیه پیش‌پردازشگر.
        
        Args:
            output_dir: مسیر ذخیره‌سازی تصاویر پیش‌پردازش شده
            target_size: اندازه هدف تصاویر
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
        # بررسی پشتیبانی GPU
        self.gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
        if self.gpu_available:
            self.logger.info("پیش‌پردازش با پشتیبانی GPU انجام خواهد شد.")
        else:
            self.logger.info("GPU در دسترس نیست، استفاده از CPU برای پیش‌پردازش.")
    
    def preprocess_image(self, image_path: Union[str, Path], grayscale: bool = True) -> np.ndarray:
        """پیش‌پردازش تصویر نمودار.
        
        Args:
            image_path: مسیر تصویر ورودی
            grayscale: تبدیل به سطح خاکستری
            
        Returns:
            آرایه نامپای تصویر پیش‌پردازش شده
        """
        image_path = Path(image_path)
        self.logger.info(f"پیش‌پردازش تصویر {image_path}...")
        
        try:
            # خواندن تصویر
            image = tf.io.read_file(str(image_path))
            
            # تبدیل به تنسور
            if str(image_path).lower().endswith(('.png', '.jpg', '.jpeg')):
                image = tf.image.decode_image(image, channels=3)
            else:
                raise ValueError(f"فرمت تصویر پشتیبانی نمی‌شود: {image_path}")
            
            # تغییر اندازه تصویر
            image = tf.image.resize(image, self.target_size)
            
            # تبدیل به سطح خاکستری
            if grayscale:
                image = tf.image.rgb_to_grayscale(image)
            
            # نرمال‌سازی
            image = image / 255.0
            
            # تبدیل به آرایه نامپای
            return image.numpy()
        
        except Exception as e:
            self.logger.error(f"خطا در پیش‌پردازش تصویر {image_path}: {e}")
            raise
    
    def batch_preprocess(self, image_paths: List[Union[str, Path]], grayscale: bool = True) -> Dict[str, np.ndarray]:
        """پیش‌پردازش دسته‌ای تصاویر.
        
        Args:
            image_paths: لیست مسیرهای تصاویر
            grayscale: تبدیل به سطح خاکستری
            
        Returns:
            دیکشنری تصاویر پیش‌پردازش شده
        """
        results = {}
        total = len(image_paths)
        
        self.logger.info(f"شروع پیش‌پردازش {total} تصویر...")
        
        for i, path in enumerate(image_paths):
            try:
                path_str = str(path)
                key = Path(path).stem
                results[key] = self.preprocess_image(path, grayscale)
                
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    self.logger.info(f"پیش‌پردازش {i + 1}/{total} تصویر تکمیل شد.")
            
            except Exception as e:
                self.logger.error(f"خطا در پیش‌پردازش تصویر {path}: {e}")
                continue
        
        return results
    
    def save_preprocessed_image(self, image: np.ndarray, filename: str) -> str:
        """ذخیره تصویر پیش‌پردازش شده.
        
        Args:
            image: آرایه تصویر
            filename: نام فایل برای ذخیره‌سازی
            
        Returns:
            مسیر فایل ذخیره شده
        """
        output_path = self.output_dir / filename
        
        # تبدیل به محدوده [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # حذف بعد کانال اضافی برای تصاویر سطح خاکستری
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.squeeze(axis=-1)
        
        # ذخیره تصویر
        cv2.imwrite(str(output_path), image)
        
        self.logger.info(f"تصویر پیش‌پردازش شده در {output_path} ذخیره شد.")
        return str(output_path)
    
    def generate_chart_from_data(self, data: pd.DataFrame, output_path: str,
                               title: Optional[str] = None, pattern_zones: Optional[List[Dict[str, Any]]] = None) -> str:
        """ایجاد تصویر نمودار از داده‌های بازار.
        
        Args:
            data: داده‌های قیمت (باید ستون‌های open, high, low, close داشته باشد)
            output_path: مسیر خروجی برای ذخیره تصویر
            title: عنوان نمودار (اختیاری)
            pattern_zones: مناطق الگو برای هایلایت کردن (اختیاری)
            
        Returns:
            مسیر تصویر ذخیره شده
        """
        self.logger.info("ایجاد تصویر نمودار از داده‌ها...")
        
        # بررسی داده‌ها
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"ستون {col} در داده‌ها وجود ندارد")
        
        # تنظیم اندازه نمودار
        plt.figure(figsize=(10, 6), dpi=100)
        
        # رسم نمودار کندل‌استیک
        ax = plt.subplot()
        
        # رنگ‌های کندل
        up_color = 'green'
        down_color = 'red'
        
        # تنظیم محور X با تاریخ
        if isinstance(data.index, pd.DatetimeIndex):
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
        
        # مشخص کردن کندل‌های صعودی و نزولی
        up = data[data.close >= data.open]
        down = data[data.close < data.open]
        
        # مقیاس کندل‌ها
        width = 0.5
        
        # رسم کندل‌های صعودی
        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color=up_color)
        ax.bar(up.index, up.high - up.close, 0.1, bottom=up.close, color=up_color)
        ax.bar(up.index, up.low - up.open, 0.1, bottom=up.open, color=up_color)
        
        # رسم کندل‌های نزولی
        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color=down_color)
        ax.bar(down.index, down.high - down.open, 0.1, bottom=down.open, color=down_color)
        ax.bar(down.index, down.low - down.close, 0.1, bottom=down.close, color=down_color)
        
        # هایلایت کردن مناطق الگو (اگر تعیین شده باشند)
        if pattern_zones:
            for zone in pattern_zones:
                start = zone.get('start')
                end = zone.get('end')
                pattern_type = zone.get('type', 'unknown')
                color = zone.get('color', 'yellow')
                alpha = zone.get('alpha', 0.3)
                
                if start and end:
                    # هایلایت کردن منطقه
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    
                    # اضافه کردن برچسب
                    if pattern_type:
                        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95, pattern_type,
                                horizontalalignment='center', color='black', fontsize=10)
        
        # تنظیم عنوان و برچسب‌ها
        if title:
            plt.title(title)
        plt.ylabel('قیمت')
        plt.grid(True, alpha=0.3)
        
        # ذخیره تصویر
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"نمودار در {output_path} ذخیره شد.")
        return output_path
    
    def apply_augmentation(self, image: np.ndarray, augmentation_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """اعمال افزایش داده بر روی تصویر.
        
        Args:
            image: تصویر ورودی
            augmentation_params: پارامترهای افزایش داده (اختیاری)
            
        Returns:
            تصویر افزایش داده شده
        """
        params = augmentation_params or {}
        
        # افزودن بعد دسته اگر نیاز باشد
        if len(image.shape) == 3 and image.shape[0] != 1:  # اگر تصویر به شکل (H, W, C) باشد
            image = tf.expand_dims(image, axis=0)  # تبدیل به (1, H, W, C)
        
        # چرخش
        if 'rotation' in params:
            angle = params['rotation']
            image = tf.image.rot90(image, k=int(angle // 90))
        
        # فلیپ افقی
        if params.get('horizontal_flip', False):
            image = tf.image.flip_left_right(image)
        
        # فلیپ عمودی
        if params.get('vertical_flip', False):
            image = tf.image.flip_up_down(image)
        
        # تغییر روشنایی
        if 'brightness' in params:
            delta = params['brightness']
            image = tf.image.adjust_brightness(image, delta)
        
        # تغییر کنتراست
        if 'contrast' in params:
            factor = params['contrast']
            image = tf.image.adjust_contrast(image, factor)
        
        # برش
        if 'crop' in params:
            crop = params['crop']
            h, w = image.shape[1:3]
            offset_h = int(crop.get('offset_height', 0) * h)
            offset_w = int(crop.get('offset_width', 0) * w)
            target_h = int(crop.get('target_height', 0.8) * h)
            target_w = int(crop.get('target_width', 0.8) * w)
            
            image = tf.image.crop_to_bounding_box(
                image, offset_h, offset_w, target_h, target_w
            )
            
            # تغییر اندازه به اندازه اصلی
            image = tf.image.resize(image, (h, w))
        
        # اضافه کردن نویز گاوسی
        if 'gaussian_noise' in params:
            std = params['gaussian_noise']
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float32)
            image = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        # حذف بعد دسته اگر اضافه کرده بودیم
        if image.shape[0] == 1:
            image = tf.squeeze(image, axis=0)
        
        return image.numpy()
    
    def generate_augmented_dataset(self, images: Dict[str, np.ndarray], num_augmentations: int = 5) -> Dict[str, np.ndarray]:
        """ایجاد مجموعه داده افزایش یافته.
        
        Args:
            images: دیکشنری تصاویر اصلی
            num_augmentations: تعداد نسخه‌های افزایش یافته برای هر تصویر
            
        Returns:
            دیکشنری تصاویر افزایش یافته
        """
        self.logger.info(f"ایجاد {num_augmentations} نسخه افزایش یافته برای {len(images)} تصویر...")
        
        augmented_images = {}
        
        for key, image in images.items():
            # اضافه کردن تصویر اصلی
            augmented_images[key] = image
            
            for i in range(num_augmentations):
                # ایجاد پارامترهای تصادفی برای افزایش داده
                aug_params = {
                    'rotation': np.random.choice([0, 90, 180, 270]),
                    'horizontal_flip': np.random.choice([True, False]),
                    'brightness': np.random.uniform(-0.2, 0.2),
                    'contrast': np.random.uniform(0.8, 1.2),
                    'gaussian_noise': np.random.uniform(0, 0.05),
                    'crop': {
                        'offset_height': np.random.uniform(0, 0.2),
                        'offset_width': np.random.uniform(0, 0.2),
                        'target_height': np.random.uniform(0.7, 0.9),
                        'target_width': np.random.uniform(0.7, 0.9)
                    }
                }
                
                # اعمال افزایش داده
                aug_image = self.apply_augmentation(image, aug_params)
                
                # اضافه کردن به مجموعه
                aug_key = f"{key}_aug_{i+1}"
                augmented_images[aug_key] = aug_image
        
        self.logger.info(f"در مجموع {len(augmented_images)} تصویر ایجاد شد.")
        return augmented_images
    
    def create_tf_dataset(self, images: Dict[str, np.ndarray], labels: Dict[str, int], 
                        batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """ایجاد دیتاست TensorFlow از تصاویر.
        
        Args:
            images: دیکشنری تصاویر
            labels: دیکشنری برچسب‌ها
            batch_size: اندازه دسته
            shuffle: مخلوط کردن داده‌ها
            
        Returns:
            دیتاست TensorFlow
        """
        # ایجاد لیست تصاویر و برچسب‌ها با ترتیب یکسان
        image_list = []
        label_list = []
        
        for key, image in images.items():
            if key in labels:
                image_list.append(image)
                label_list.append(labels[key])
        
        # تبدیل به تنسور
        X = tf.convert_to_tensor(np.array(image_list))
        y = tf.convert_to_tensor(np.array(label_list))
        
        # ایجاد دیتاست
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_list))
        
        # دسته‌بندی و پیش‌بارگیری
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset 