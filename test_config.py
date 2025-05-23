"""
اسکریپت تست برای ماژول ConfigManager
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from findataanalyzer.utils.config import ConfigManager

def main():
    """تست ماژول ConfigManager."""
    # ایجاد تمام مسیرهای مورد نیاز
    os.makedirs('logs', exist_ok=True)
    os.makedirs('experiments/metrics', exist_ok=True)
    os.makedirs('experiments/tensorboard', exist_ok=True)
    os.makedirs('experiments/log', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    
    try:
        # بارگذاری کانفیگ
        print("بارگذاری تنظیمات از configs/settings.yaml...")
        config_manager = ConfigManager("configs/settings.yaml")
        
        # تست دسترسی به تنظیمات
        print("\nتست دسترسی به تنظیمات:")
        server_host = config_manager.get("server.host")
        print(f"server.host: {server_host}")
        
        # تست تغییر تنظیمات
        print("\nتست تغییر تنظیمات:")
        config_manager.set("server.debug", False)
        print(f"server.debug: {config_manager.get('server.debug')}")
        
        print("\nتنظیمات با موفقیت بارگذاری و تست شد.")
    except Exception as e:
        print(f"خطا در تست تنظیمات: {e}")

if __name__ == "__main__":
    main() 