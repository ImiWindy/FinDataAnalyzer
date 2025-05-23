"""
اسکریپت تست API
"""

import sys
import os
import argparse

# اضافه کردن مسیر src به PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import uvicorn
from fastapi.testclient import TestClient
from findataanalyzer.api.main import app

def main():
    """تابع اصلی برای اجرای تست API."""
    parser = argparse.ArgumentParser(description="تست API")
    parser.add_argument('--run-server', action='store_true',
                      help='اجرای سرور API')
    parser.add_argument('--port', type=int, default=8000,
                      help='پورت سرور API')
    args = parser.parse_args()
    
    if args.run_server:
        # اجرای سرور
        print(f"راه‌اندازی سرور API روی پورت {args.port}...")
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        # تست API بدون اجرای سرور
        client = TestClient(app)
        
        print("تست endpoint اصلی...")
        response = client.get("/")
        print(f"پاسخ: {response.status_code} - {response.json()}")
        
        print("تست endpoint سلامت...")
        response = client.get("/health")
        print(f"پاسخ: {response.status_code} - {response.json()}")
        
        print("تست endpoint تحلیل...")
        test_data = {
            "data_source": "data/samples/test",
            "parameters": {
                "advanced": True
            }
        }
        response = client.post("/api/v1/analysis/analyze", json=test_data)
        print(f"پاسخ تحلیل: {response.status_code}")
        if response.status_code == 200:
            print("تست API با موفقیت انجام شد.")
        else:
            print(f"خطا در تست API: {response.json()}")

if __name__ == "__main__":
    main() 