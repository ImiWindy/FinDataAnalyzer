"""
اسکریپت تست ارزیابی مدل
"""

import sys
import os
import argparse

# اضافه کردن مسیر src به PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from findataanalyzer.evaluate import load_model, load_test_data, evaluate_model, visualize_results, save_results

def main():
    """تابع اصلی برای اجرای تست ارزیابی."""
    parser = argparse.ArgumentParser(description="تست ارزیابی مدل")
    parser.add_argument('--model', type=str, default='models/test/model.h5',
                      help='مسیر مدل برای ارزیابی')
    parser.add_argument('--data', type=str, default='data/samples/test',
                      help='مسیر داده‌های تست')
    parser.add_argument('--output', type=str, default='results/evaluation',
                      help='مسیر خروجی نتایج')
    parser.add_argument('--visualize', action='store_true',
                      help='تولید تجسم‌های نتایج')
    args = parser.parse_args()
    
    # ایجاد دایرکتوری خروجی
    os.makedirs(args.output, exist_ok=True)
    
    try:
        print(f"بارگذاری مدل از {args.model}...")
        model = load_model(args.model)
        
        print(f"بارگذاری داده‌های تست از {args.data}...")
        X, y, class_names, image_paths = load_test_data(args.data)
        
        print("ارزیابی مدل...")
        results = evaluate_model(model, X, y, class_names)
        
        print(f"دقت مدل: {results['accuracy']:.4f}")
        
        save_results(results, args.output)
        print(f"نتایج در {args.output} ذخیره شد.")
        
        if args.visualize:
            print("ایجاد تجسم‌های نتایج...")
            visualize_results(results, class_names, X, y, args.output)
            print(f"تجسم‌ها در {args.output} ذخیره شد.")
        
        print("تست ارزیابی با موفقیت انجام شد.")
    except Exception as e:
        print(f"خطا در تست ارزیابی: {e}")
        raise

if __name__ == "__main__":
    main() 