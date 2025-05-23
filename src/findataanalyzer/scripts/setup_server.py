#!/usr/bin/env python
"""اسکریپت راه‌اندازی سرور محاسباتی.

این اسکریپت محیط اجرایی سیستم را تنظیم کرده و سرویس‌های مورد نیاز را راه‌اندازی می‌کند.
"""

import os
import sys
import argparse
import logging
import subprocess
import platform
from pathlib import Path
import yaml
import shutil
import time

# تنظیم مسیر برای واردسازی ماژول‌های پروژه
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from findataanalyzer.utils.config import ConfigManager
from findataanalyzer.utils.logger import LogManager


def parse_args():
    """تجزیه آرگومان‌های خط فرمان."""
    parser = argparse.ArgumentParser(description="راه‌اندازی سرور محاسباتی")
    parser.add_argument(
        "--config", type=str, default="configs/settings.yaml",
        help="مسیر فایل تنظیمات"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="فعال‌سازی پشتیبانی از GPU"
    )
    parser.add_argument(
        "--env", type=str, choices=["dev", "prod", "test"], default="dev",
        help="محیط اجرایی (dev/prod/test)"
    )
    parser.add_argument(
        "--setup-db", action="store_true", default=False,
        help="راه‌اندازی پایگاه داده"
    )
    parser.add_argument(
        "--setup-dirs", action="store_true", default=False,
        help="ایجاد ساختار دایرکتوری‌ها"
    )
    parser.add_argument(
        "--install-deps", action="store_true", default=False,
        help="نصب وابستگی‌های ضروری"
    )
    return parser.parse_args()


def check_gpu_support():
    """بررسی پشتیبانی از GPU."""
    print("بررسی پشتیبانی از GPU...")
    
    # بررسی TensorFlow با CUDA
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow با پشتیبانی GPU پیدا شد (نسخه: {tf.__version__})")
            print(f"  - دستگاه‌های یافت شده: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  - {i}: {gpu.name}")
            return True
        else:
            print("❌ TensorFlow با پشتیبانی GPU پیدا نشد")
    except ImportError:
        print("❌ TensorFlow نصب نشده است")
    except Exception as e:
        print(f"❌ خطا در بررسی GPU: {e}")
    
    return False


def setup_directories(config):
    """ایجاد ساختار دایرکتوری‌ها."""
    print("ایجاد ساختار دایرکتوری‌ها...")
    
    # دایرکتوری‌های اصلی
    directories = [
        config.get("data_pipeline.raw_data_dir", "data/raw"),
        config.get("data_pipeline.processed_data_dir", "data/processed"),
        config.get("data_pipeline.cache_dir", "data/cache"),
        config.get("experiment_tracking.experiments_dir", "experiments"),
        config.get("experiment_tracking.metrics_dir", "experiments/metrics"),
        config.get("experiment_tracking.checkpoints_dir", "experiments/checkpoints"),
        config.get("experiment_tracking.tensorboard_dir", "experiments/tensorboard"),
        "logs",
        "configs"
    ]
    
    # ایجاد دایرکتوری‌ها
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ دایرکتوری {path} ایجاد شد")


def setup_database(config):
    """راه‌اندازی پایگاه داده."""
    print("راه‌اندازی پایگاه داده...")
    
    db_url = config.get("database.url", "sqlite:///data/findataanalyzer.db")
    
    if db_url.startswith("sqlite"):
        # استخراج مسیر فایل SQLite
        import re
        match = re.search(r"sqlite:///(.+)", db_url)
        if match:
            db_path = match.group(1)
            db_dir = os.path.dirname(db_path)
            
            # ایجاد دایرکتوری
            Path(db_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ دایرکتوری پایگاه داده در {db_dir} ایجاد شد")
            
            # بررسی اگر فایل وجود داشته باشد
            if os.path.exists(db_path):
                print(f"📝 فایل پایگاه داده از قبل در {db_path} وجود دارد")
            else:
                # ایجاد پایگاه داده خالی
                try:
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    conn.close()
                    print(f"✅ فایل پایگاه داده SQLite در {db_path} ایجاد شد")
                except Exception as e:
                    print(f"❌ خطا در ایجاد پایگاه داده: {e}")
    
    # ایجاد جداول پایگاه داده
    try:
        print("ایجاد جداول پایگاه داده...")
        from findataanalyzer.utils.database import DatabaseManager
        db_manager = DatabaseManager(config.get("database", {}))
        print("✅ جداول پایگاه داده ایجاد شدند")
    except Exception as e:
        print(f"❌ خطا در ایجاد جداول پایگاه داده: {e}")


def install_dependencies(gpu_support=False):
    """نصب وابستگی‌های ضروری."""
    print("نصب وابستگی‌های ضروری...")
    
    # لیست وابستگی‌های پایه
    base_packages = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "requests",
        "schedule",
        "sqlalchemy",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pillow",
        "opencv-python"
    ]
    
    # وابستگی‌های GPU
    gpu_packages = [
        "tensorflow",
        "tensorboard"
    ]
    
    # انتخاب لیست نهایی
    packages = base_packages
    if gpu_support:
        packages.extend(gpu_packages)
    
    # نصب وابستگی‌ها
    try:
        print(f"نصب {len(packages)} پکیج...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ وابستگی‌ها با موفقیت نصب شدند")
    except subprocess.CalledProcessError as e:
        print(f"❌ خطا در نصب وابستگی‌ها: {e}")
        if e.stderr:
            print(f"خطای خروجی: {e.stderr.decode('utf-8')}")


def copy_default_config():
    """کپی فایل تنظیمات پیش‌فرض."""
    print("بررسی وجود فایل تنظیمات...")
    
    config_path = Path("configs/settings.yaml")
    default_config_path = Path(__file__).parent / "default_settings.yaml"
    
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if default_config_path.exists():
            shutil.copy(default_config_path, config_path)
            print(f"✅ فایل تنظیمات پیش‌فرض در {config_path} کپی شد")
        else:
            # ایجاد فایل تنظیمات پیش‌فرض
            default_config = {
                "gpu": {
                    "enabled": False,
                    "device": "cuda:0",
                    "memory_fraction": 0.8,
                    "allow_growth": True
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4,
                    "timeout": 300,
                    "max_requests": 1000
                },
                "database": {
                    "url": "sqlite:///data/findataanalyzer.db",
                    "pool_size": 5,
                    "pool_recycle": 3600
                },
                "data_pipeline": {
                    "raw_data_dir": "data/raw",
                    "processed_data_dir": "data/processed",
                    "cache_dir": "data/cache",
                    "batch_size": 32,
                    "num_workers": 4
                },
                "experiment_tracking": {
                    "enabled": True,
                    "backend": "sqlite",
                    "experiments_dir": "experiments",
                    "metrics_dir": "experiments/metrics",
                    "checkpoints_dir": "experiments/checkpoints",
                    "tensorboard_dir": "experiments/tensorboard"
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file": "logs/findataanalyzer.log",
                    "max_bytes": 10485760,
                    "backup_count": 5
                },
                "model": {
                    "input_size": [224, 224],
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "early_stopping_patience": 10,
                    "val_split": 0.15,
                    "test_split": 0.15,
                    "models_dir": "models",
                    "checkpoints_dir": "models/checkpoints"
                }
            }
            
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"✅ فایل تنظیمات پیش‌فرض در {config_path} ایجاد شد")
    else:
        print(f"📝 فایل تنظیمات در {config_path} از قبل وجود دارد")


def update_config_with_gpu(config_path, gpu_enabled):
    """به‌روزرسانی تنظیمات GPU در فایل پیکربندی."""
    if not os.path.exists(config_path):
        print(f"❌ فایل تنظیمات در {config_path} یافت نشد")
        return
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if "gpu" not in config:
            config["gpu"] = {}
        
        config["gpu"]["enabled"] = gpu_enabled
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ تنظیمات GPU در فایل {config_path} به‌روزرسانی شد")
    except Exception as e:
        print(f"❌ خطا در به‌روزرسانی تنظیمات GPU: {e}")


def start_api_server(config):
    """راه‌اندازی سرور API."""
    print("راه‌اندازی سرور API...")
    
    host = config.get("server.host", "0.0.0.0")
    port = config.get("server.port", 8000)
    workers = config.get("server.workers", 4)
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "findataanalyzer.api.main:app", 
            "--host", host, 
            "--port", str(port),
            "--workers", str(workers)
        ]
        
        print(f"در حال راه‌اندازی سرور API در http://{host}:{port}")
        process = subprocess.Popen(cmd)
        
        # کمی صبر کنیم تا سرور راه‌اندازی شود
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ سرور API با موفقیت در http://{host}:{port} راه‌اندازی شد")
            return process
        else:
            print("❌ خطا در راه‌اندازی سرور API")
            return None
    except Exception as e:
        print(f"❌ خطا در راه‌اندازی سرور API: {e}")
        return None


def main():
    """تابع اصلی."""
    args = parse_args()
    
    print("=" * 60)
    print(" راه‌اندازی سرور محاسباتی FinDataAnalyzer ")
    print("=" * 60)
    
    # بررسی و ایجاد فایل تنظیمات
    copy_default_config()
    
    # بارگذاری تنظیمات
    config_manager = ConfigManager(args.config)
    
    # به‌روزرسانی تنظیمات GPU
    if args.gpu:
        gpu_supported = check_gpu_support()
        update_config_with_gpu(args.config, gpu_supported)
    
    # ایجاد ساختار دایرکتوری‌ها
    if args.setup_dirs:
        setup_directories(config_manager.get_all())
    
    # راه‌اندازی پایگاه داده
    if args.setup_db:
        setup_database(config_manager.get_all())
    
    # نصب وابستگی‌ها
    if args.install_deps:
        install_dependencies(args.gpu)
    
    # راه‌اندازی LogManager
    log_config = config_manager.get("logging", {})
    log_manager = LogManager({"logging": log_config})
    logger = log_manager.get_logger()
    
    # راه‌اندازی سرور API (فقط در محیط تولید)
    if args.env == "prod":
        api_server = start_api_server(config_manager.get_all())
    
    logger.info("راه‌اندازی سرور محاسباتی با موفقیت انجام شد")
    print("\n✅ راه‌اندازی سرور محاسباتی با موفقیت انجام شد")


if __name__ == "__main__":
    main() 