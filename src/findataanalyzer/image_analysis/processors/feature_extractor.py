"""ماژول استخراج ویژگی از تصاویر نمودارهای مالی.

این ماژول مسئول استخراج ویژگی‌های پیشرفته از تصاویر نمودارهای مالی است.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os
import logging
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from findataanalyzer.image_analysis.processors.image_processor import ImageProcessor


class FeatureExtractor:
    """کلاس استخراج ویژگی از تصاویر."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        مقداردهی اولیه استخراج‌کننده ویژگی.
        
        Args:
            config: تنظیمات استخراج ویژگی (اختیاری)
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # تنظیم دستگاه محاسباتی
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"دستگاه محاسباتی: {self.device}")
        
        # تنظیم تبدیلات تصویر
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # لود کردن مدل‌های پیش‌آموزش دیده
        self._load_models()
    
    def _load_models(self) -> None:
        """
        لود کردن مدل‌های پیش‌آموزش دیده برای استخراج ویژگی.
        """
        self.logger.info("لود کردن مدل‌های پیش‌آموزش دیده برای استخراج ویژگی...")
        
        try:
            # ResNet-50
            self.resnet = models.resnet50(pretrained=True)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # حذف لایه طبقه‌بندی
            self.resnet = self.resnet.to(self.device)
            self.resnet.eval()
            
            # VGG-16
            self.vgg = models.vgg16(pretrained=True).features  # فقط لایه‌های استخراج ویژگی
            self.vgg = self.vgg.to(self.device)
            self.vgg.eval()
            
            # MobileNetV2
            self.mobilenet = models.mobilenet_v2(pretrained=True).features
            self.mobilenet = self.mobilenet.to(self.device)
            self.mobilenet.eval()
            
            self.logger.info("لود کردن مدل‌ها با موفقیت انجام شد")
        except Exception as e:
            self.logger.error(f"خطا در لود کردن مدل‌ها: {e}")
            raise RuntimeError(f"خطا در لود کردن مدل‌ها: {e}")
    
    def extract_deep_features(self, image_path: str, model_name: str = "resnet") -> np.ndarray:
        """
        استخراج ویژگی‌های عمیق از تصویر با استفاده از شبکه‌های پیش‌آموزش دیده.
        
        Args:
            image_path: مسیر تصویر
            model_name: نام مدل (resnet, vgg یا mobilenet)
            
        Returns:
            بردار ویژگی
        """
        try:
            # خواندن و پیش‌پردازش تصویر
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # استخراج ویژگی با مدل منتخب
            with torch.no_grad():
                if model_name == "resnet":
                    features = self.resnet(image_tensor)
                elif model_name == "vgg":
                    features = self.vgg(image_tensor)
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                elif model_name == "mobilenet":
                    features = self.mobilenet(image_tensor)
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                else:
                    raise ValueError(f"مدل نامعتبر: {model_name}")
            
            # تغییر شکل به بردار یک‌بعدی
            features = features.squeeze().flatten().cpu().numpy()
            return features
        
        except Exception as e:
            self.logger.error(f"خطا در استخراج ویژگی‌های عمیق: {e}")
            raise RuntimeError(f"خطا در استخراج ویژگی‌های عمیق: {e}")
    
    def extract_traditional_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        استخراج ویژگی‌های سنتی از تصویر (SIFT, HOG, و غیره).
        
        Args:
            image_path: مسیر تصویر
            
        Returns:
            دیکشنری ویژگی‌ها
        """
        try:
            # خواندن تصویر
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"خطا در خواندن تصویر: {image_path}")
            
            # تبدیل به سطح خاکستری
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = {}
            
            # استخراج ویژگی‌های HOG
            hog = self._extract_hog_features(gray)
            features['hog'] = hog
            
            # استخراج ویژگی‌های LBP
            lbp = self._extract_lbp_features(gray)
            features['lbp'] = lbp
            
            # استخراج ویژگی‌های هیستوگرام
            hist_features = self._extract_histogram_features(image)
            features['histogram'] = hist_features
            
            # محاسبه آمار تصویر
            stats_features = self._extract_statistical_features(gray)
            features['stats'] = stats_features
            
            return features
        
        except Exception as e:
            self.logger.error(f"خطا در استخراج ویژگی‌های سنتی: {e}")
            raise RuntimeError(f"خطا در استخراج ویژگی‌های سنتی: {e}")
    
    def _extract_hog_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        استخراج ویژگی‌های HOG از تصویر.
        
        Args:
            gray_image: تصویر سطح خاکستری
            
        Returns:
            ویژگی‌های HOG
        """
        # تغییر اندازه به سایز استاندارد
        resized = cv2.resize(gray_image, (64, 64))
        
        # استخراج ویژگی‌های HOG
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        return features.flatten()
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        استخراج ویژگی‌های LBP از تصویر.
        
        Args:
            gray_image: تصویر سطح خاکستری
            
        Returns:
            ویژگی‌های LBP
        """
        from skimage.feature import local_binary_pattern
        
        # تغییر اندازه به سایز استاندارد
        resized = cv2.resize(gray_image, (64, 64))
        
        # پارامترهای LBP
        radius = 3
        n_points = 8 * radius
        
        # محاسبه LBP
        lbp = local_binary_pattern(resized, n_points, radius, method='uniform')
        
        # ایجاد هیستوگرام از LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def _extract_histogram_features(self, image: np.ndarray) -> np.ndarray:
        """
        استخراج ویژگی‌های هیستوگرام از تصویر.
        
        Args:
            image: تصویر
            
        Returns:
            ویژگی‌های هیستوگرام
        """
        # تغییر اندازه به سایز استاندارد
        resized = cv2.resize(image, (64, 64))
        
        # جداسازی کانال‌های رنگی
        b, g, r = cv2.split(resized)
        
        # محاسبه هیستوگرام هر کانال
        hist_b = cv2.calcHist([b], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [16], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [16], [0, 256])
        
        # نرمال‌سازی هیستوگرام‌ها
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        # ترکیب هیستوگرام‌ها
        hist_features = np.concatenate([hist_b, hist_g, hist_r])
        
        return hist_features
    
    def _extract_statistical_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        استخراج ویژگی‌های آماری از تصویر.
        
        Args:
            gray_image: تصویر سطح خاکستری
            
        Returns:
            ویژگی‌های آماری
        """
        # محاسبه آمار
        mean = np.mean(gray_image)
        std = np.std(gray_image)
        skewness = np.mean(((gray_image - mean) / (std + 1e-10)) ** 3)
        kurtosis = np.mean(((gray_image - mean) / (std + 1e-10)) ** 4) - 3
        
        # محاسبه لبه‌ها
        edges = cv2.Canny(gray_image, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # ایجاد بردار ویژگی
        stats_features = np.array([mean, std, skewness, kurtosis, edge_density])
        
        return stats_features
    
    def detect_candlestick_patterns(self, image_path: str) -> Dict[str, float]:
        """
        تشخیص الگوهای شمعی در نمودارهای مالی.
        
        Args:
            image_path: مسیر تصویر
            
        Returns:
            دیکشنری الگوهای تشخیص داده شده و احتمال آنها
        """
        # این متد یک پیاده‌سازی پایه است و می‌تواند با استفاده از 
        # الگوریتم‌های تشخیص تصویر پیشرفته‌تر بهبود یابد
        try:
            # استخراج ویژگی‌های عمیق
            deep_features = self.extract_deep_features(image_path)
            
            # در این مرحله، یک مدل طبقه‌بندی آموزش دیده باید بر روی ویژگی‌ها اعمال شود
            # این قسمت به آموزش یک طبقه‌بند نیاز دارد
            
            # مقادیر نمونه برای نمایش عملکرد
            patterns = {
                "doji": 0.15,
                "hammer": 0.25,
                "engulfing_bullish": 0.35,
                "engulfing_bearish": 0.05,
                "morning_star": 0.10,
                "evening_star": 0.05,
                "shooting_star": 0.05
            }
            
            return patterns
        
        except Exception as e:
            self.logger.error(f"خطا در تشخیص الگوهای شمعی: {e}")
            return {}
    
    def detect_chart_patterns(self, image_path: str) -> Dict[str, float]:
        """
        تشخیص الگوهای نمودار در تصاویر مالی.
        
        Args:
            image_path: مسیر تصویر
            
        Returns:
            دیکشنری الگوهای تشخیص داده شده و احتمال آنها
        """
        try:
            # استخراج ویژگی‌های عمیق
            deep_features = self.extract_deep_features(image_path)
            
            # در این مرحله، یک مدل طبقه‌بندی آموزش دیده باید بر روی ویژگی‌ها اعمال شود
            
            # مقادیر نمونه برای نمایش عملکرد
            patterns = {
                "head_and_shoulders": 0.15,
                "double_top": 0.25,
                "double_bottom": 0.35,
                "triangle": 0.05,
                "flag": 0.10,
                "channel": 0.05,
                "cup_and_handle": 0.05
            }
            
            return patterns
        
        except Exception as e:
            self.logger.error(f"خطا در تشخیص الگوهای نمودار: {e}")
            return {}
    
    def detect_trend_lines(self, image_path: str) -> Dict[str, Any]:
        """
        تشخیص خطوط روند در نمودارهای مالی.
        
        Args:
            image_path: مسیر تصویر
            
        Returns:
            دیکشنری خطوط روند تشخیص داده شده
        """
        try:
            # خواندن تصویر
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"خطا در خواندن تصویر: {image_path}")
            
            # تبدیل به سطح خاکستری
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # تشخیص لبه‌ها
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # تشخیص خطوط با تبدیل هاف
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            # فیلتر کردن خطوط برای یافتن خطوط روند احتمالی
            trend_lines = []
            if lines is not None:
                for i in range(min(len(lines), 10)):  # حداکثر 10 خط
                    rho, theta = lines[i][0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    
                    # فیلتر کردن خطوط عمودی و افقی
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        trend_lines.append({
                            "start": (x1, y1),
                            "end": (x2, y2),
                            "angle": np.degrees(theta),
                            "confidence": 0.5  # مقدار نمونه
                        })
            
            # محاسبه روند کلی بر اساس خطوط تشخیص داده شده
            uptrend_count = 0
            downtrend_count = 0
            for line in trend_lines:
                if line["start"][1] > line["end"][1]:
                    uptrend_count += 1
                else:
                    downtrend_count += 1
            
            trend_direction = "uptrend" if uptrend_count > downtrend_count else "downtrend"
            trend_strength = abs(uptrend_count - downtrend_count) / max(1, len(trend_lines))
            
            return {
                "trend_lines": trend_lines,
                "trend_direction": trend_direction,
                "trend_strength": float(trend_strength),
                "line_count": len(trend_lines)
            }
        
        except Exception as e:
            self.logger.error(f"خطا در تشخیص خطوط روند: {e}")
            return {"trend_lines": [], "trend_direction": "unknown", "trend_strength": 0.0}
    
    def extract_features_batch(self, image_paths: List[str], include_deep: bool = True, 
                              include_traditional: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        استخراج ویژگی‌ها از یک دسته تصاویر.
        
        Args:
            image_paths: لیست مسیرهای تصاویر
            include_deep: شامل کردن ویژگی‌های عمیق
            include_traditional: شامل کردن ویژگی‌های سنتی
            
        Returns:
            دیکشنری ویژگی‌های تصاویر
        """
        features_batch = {}
        
        for image_path in image_paths:
            try:
                features = {}
                
                if include_deep:
                    features["resnet"] = self.extract_deep_features(image_path, "resnet")
                    features["vgg"] = self.extract_deep_features(image_path, "vgg")
                    features["mobilenet"] = self.extract_deep_features(image_path, "mobilenet")
                
                if include_traditional:
                    traditional_features = self.extract_traditional_features(image_path)
                    features.update(traditional_features)
                
                features_batch[image_path] = features
                
            except Exception as e:
                self.logger.error(f"خطا در استخراج ویژگی‌های تصویر {image_path}: {e}")
        
        return features_batch 