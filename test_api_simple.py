"""
اسکریپت تست ساده API
"""

import sys
import os

# اضافه کردن مسیر src به PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from findataanalyzer.api.main import app
from fastapi.testclient import TestClient

# ایجاد یک نمونه تست کلاینت
client = TestClient(app)

def test_root_endpoint():
    """تست endpoint اصلی."""
    response = client.get("/")
    print(f"پاسخ endpoint اصلی: {response.status_code}")
    if response.status_code == 200:
        print(f"محتوا: {response.json()}")
    return response.status_code == 200

def test_health_endpoint():
    """تست endpoint سلامت."""
    response = client.get("/health")
    print(f"پاسخ endpoint سلامت: {response.status_code}")
    if response.status_code == 200:
        print(f"محتوا: {response.json()}")
    return response.status_code == 200

def main():
    """تابع اصلی."""
    print("شروع تست‌های ساده API...")
    
    root_test_passed = test_root_endpoint()
    health_test_passed = test_health_endpoint()
    
    if root_test_passed and health_test_passed:
        print("تمام تست‌ها با موفقیت انجام شدند!")
    else:
        print("برخی از تست‌ها ناموفق بودند.")

if __name__ == "__main__":
    main() 