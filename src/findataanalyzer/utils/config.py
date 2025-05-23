"""ماژول مدیریت تنظیمات.

این ماژول امکان بارگذاری و استفاده از تنظیمات پروژه را فراهم می‌کند.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


def get_config_path() -> Path:
    """دریافت مسیر فایل تنظیمات بر اساس متغیر محیطی یا مقدار پیش‌فرض.
    
    Returns:
        مسیر فایل تنظیمات
    """
    env_path = os.environ.get("FINDATAANALYZER_CONFIG", "configs/settings.yaml")
    return Path(env_path)


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """بارگذاری تنظیمات از فایل.
    
    Args:
        config_path: مسیر فایل تنظیمات (اختیاری، اگر مشخص نشود از تابع get_config_path استفاده می‌شود)
        
    Returns:
        دیکشنری تنظیمات
    """
    if config_path is None:
        config_path = get_config_path()
    
    if not config_path.exists():
        raise FileNotFoundError(f"فایل تنظیمات در مسیر {config_path} یافت نشد")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """دریافت تنظیمات پیش‌فرض.
    
    Returns:
        دیکشنری تنظیمات پیش‌فرض
    """
    return {
        "gpu": {
            "use_gpu": False,
            "memory_limit": 0.5,
            "allow_growth": True,
            "visible_devices": "0"
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/app.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "model": {
            "type": "default",
            "input_shape": [224, 224, 3],
            "learning_rate": 0.001,
            "dropout_rate": 0.5
        }
    }


def get_environment_config() -> Dict[str, Any]:
    """دریافت تنظیمات از متغیرهای محیطی.
    
    Returns:
        دیکشنری تنظیمات محیطی
    """
    env_config = {}
    
    # دریافت تنظیمات پایگاه داده
    db_url = os.environ.get("FINDATAANALYZER_DB_URL")
    if db_url:
        env_config["database"] = {"url": db_url}
    
    # دریافت سطح لاگ
    log_level = os.environ.get("FINDATAANALYZER_LOG_LEVEL")
    if log_level:
        if "logging" not in env_config:
            env_config["logging"] = {}
        env_config["logging"]["level"] = log_level
    
    return env_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """ترکیب تنظیمات پایه با تنظیمات برتر.
    
    Args:
        base_config: تنظیمات پایه
        override_config: تنظیمات برتر که اولویت دارند
        
    Returns:
        تنظیمات ترکیب شده
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # اگر هر دو مقدار دیکشنری هستند، ترکیب بازگشتی
            merged[key] = merge_configs(merged[key], value)
        else:
            # در غیر این صورت، مقدار جدید را جایگزین کن
            merged[key] = value
    
    return merged


def load_config() -> Dict[str, Any]:
    """بارگذاری تنظیمات از فایل، محیط و مقادیر پیش‌فرض.
    
    Returns:
        تنظیمات ترکیب شده نهایی
    """
    default_config = get_default_config()
    
    try:
        file_config = get_config()
    except FileNotFoundError:
        file_config = {}
    
    env_config = get_environment_config()
    
    # ترکیب تنظیمات با اولویت:
    # 1. متغیرهای محیطی 
    # 2. فایل تنظیمات
    # 3. تنظیمات پیش‌فرض
    merged_config = merge_configs(default_config, file_config)
    merged_config = merge_configs(merged_config, env_config)
    
    return merged_config


class ConfigManager:
    """کلاس مدیریت تنظیمات."""
    
    def __init__(self, config_path: str = "configs/settings.yaml"):
        """مقداردهی اولیه کلاس مدیریت تنظیمات.
        
        Args:
            config_path: مسیر فایل تنظیمات
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """بارگذاری تنظیمات از فایل."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # ایجاد مسیرهای مورد نیاز
            self._setup_paths()
        except Exception as e:
            raise RuntimeError(f"خطا در بارگذاری تنظیمات: {str(e)}")
    
    def _setup_paths(self) -> None:
        """ایجاد مسیرهای مورد نیاز بر اساس تنظیمات."""
        try:
            # ایجاد مسیر لاگ
            if "logging" in self.config and "file_path" in self.config["logging"]:
                log_path = Path(self.config["logging"]["file_path"])
                os.makedirs(log_path.parent, exist_ok=True)
            
            # ایجاد مسیرهای آزمایش
            if "experiment_tracking" in self.config:
                # تضمین وجود تمام کلیدهای مورد نیاز
                required_dirs = [
                    "base_dir", 
                    "checkpoints_dir", 
                    "experiments_dir", 
                    "metrics_dir", 
                    "tensorboard_dir"
                ]
                
                tracking_config = self.config["experiment_tracking"]
                for dir_key in required_dirs:
                    if dir_key in tracking_config:
                        dir_path = Path(tracking_config[dir_key])
                        os.makedirs(dir_path, exist_ok=True)
            
            # ایجاد مسیرهای داده
            if "data_pipeline" in self.config:
                data_config = self.config["data_pipeline"]
                for key in ["raw_data_dir", "processed_data_dir", "cache_dir", "image_dir"]:
                    if key in data_config:
                        dir_path = Path(data_config[key])
                        os.makedirs(dir_path, exist_ok=True)
            
            # ایجاد مسیر پایگاه داده
            if "database" in self.config and "url" in self.config["database"]:
                url = self.config["database"]["url"]
                if url.startswith("sqlite:///"):
                    db_path = Path(url.replace("sqlite:///", ""))
                    os.makedirs(db_path.parent, exist_ok=True)
                    
        except Exception as e:
            raise RuntimeError(f"خطا در راه‌اندازی مسیرها: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """دریافت مقدار یک تنظیم.
        
        Args:
            key: کلید تنظیم (می‌تواند به صورت 'parent.child' باشد)
            default: مقدار پیش‌فرض در صورت عدم وجود کلید
            
        Returns:
            مقدار تنظیم یا مقدار پیش‌فرض
        """
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
            return current
        else:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """تنظیم مقدار یک تنظیم.
        
        Args:
            key: کلید تنظیم (می‌تواند به صورت 'parent.child' باشد)
            value: مقدار جدید
        """
        if "." in key:
            parts = key.split(".")
            current = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """ذخیره تنظیمات به فایل.
        
        Args:
            path: مسیر فایل برای ذخیره (اختیاری، اگر مشخص نشود از self.config_path استفاده می‌شود)
        """
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get_all(self) -> Dict[str, Any]:
        """دریافت تمام تنظیمات.
        
        Returns:
            دیکشنری کامل تنظیمات
        """
        return self.config.copy() 