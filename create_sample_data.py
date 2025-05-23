"""
اسکریپت ایجاد داده‌های نمونه برای تست
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path

# ایجاد دایرکتوری‌های مورد نیاز
os.makedirs('data/samples/test/double_top', exist_ok=True)
os.makedirs('data/samples/test/double_bottom', exist_ok=True)
os.makedirs('data/samples/test/head_and_shoulders', exist_ok=True)
os.makedirs('data/samples/test/inverse_head_and_shoulders', exist_ok=True)
os.makedirs('data/samples/test/triangle', exist_ok=True)

# مسیر برای ذخیره تصاویر
base_path = Path('data/samples/test')

# تعداد تصاویر برای هر کلاس
n_samples = 5

# کلاس‌ها و برچسب‌ها
classes = ["double_top", "double_bottom", "head_and_shoulders", "inverse_head_and_shoulders", "triangle"]

# ایجاد داده برای metadata
metadata = {
    "classes": classes,
    "images": []
}

# ایجاد تصاویر ساختگی برای هر کلاس
for label, class_name in enumerate(classes):
    for i in range(n_samples):
        # ایجاد تصویر ساختگی با اشکال مختلف
        img = np.ones((224, 224, 3), dtype=np.uint8) * 240
        
        if class_name == "double_top":
            # ایجاد double top pattern
            x = np.linspace(0, 223, 224)
            y1 = 200 - 50 * np.sin(x/20) - 50 * np.exp(-((x-60)**2)/900)
            y2 = 200 - 50 * np.sin(x/20) - 50 * np.exp(-((x-160)**2)/900)
            
            # رسم منحنی 
            pts = np.array([[int(x[j]), int(min(y1[j], y2[j]))] for j in range(len(x))], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 0, 255), 2)
            
        elif class_name == "double_bottom":
            # ایجاد double bottom pattern
            x = np.linspace(0, 223, 224)
            y1 = 100 + 50 * np.sin(x/20) + 50 * np.exp(-((x-60)**2)/900)
            y2 = 100 + 50 * np.sin(x/20) + 50 * np.exp(-((x-160)**2)/900)
            
            # رسم منحنی
            pts = np.array([[int(x[j]), int(max(y1[j], y2[j]))] for j in range(len(x))], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 0), 2)
            
        elif class_name == "head_and_shoulders":
            # ایجاد head and shoulders pattern
            x = np.linspace(0, 223, 224)
            y = 150 - 30 * np.exp(-((x-50)**2)/200) - 60 * np.exp(-((x-112)**2)/200) - 30 * np.exp(-((x-174)**2)/200)
            
            # رسم منحنی
            pts = np.array([[int(x[j]), int(y[j])] for j in range(len(x))], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 0, 255), 2)
            
        elif class_name == "inverse_head_and_shoulders":
            # ایجاد inverse head and shoulders pattern
            x = np.linspace(0, 223, 224)
            y = 150 + 30 * np.exp(-((x-50)**2)/200) + 60 * np.exp(-((x-112)**2)/200) + 30 * np.exp(-((x-174)**2)/200)
            
            # رسم منحنی
            pts = np.array([[int(x[j]), int(y[j])] for j in range(len(x))], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 0), 2)
            
        elif class_name == "triangle":
            # ایجاد triangle pattern
            height = np.random.randint(100, 150)
            p1 = (50, 180)
            p2 = (120, 180 - height)
            p3 = (180, 180)
            
            # رسم مثلث
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.line(img, p2, p3, (0, 0, 255), 2)
            cv2.line(img, p3, p1, (0, 0, 255), 2)
        
        # اضافه کردن کمی نویز به تصویر
        noise = np.random.randint(0, 30, size=(224, 224, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # ذخیره تصویر
        filename = f"{class_name}_sample_{i}.png"
        file_path = base_path / class_name / filename
        cv2.imwrite(str(file_path), img)
        
        # افزودن اطلاعات تصویر به metadata
        metadata["images"].append({
            "filename": f"{class_name}/{filename}",
            "label": label
        })
        
        print(f"تصویر {filename} ایجاد شد.")

# ذخیره metadata
with open(base_path / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nتمام {n_samples * len(classes)} تصویر نمونه با موفقیت ایجاد شد.")
print(f"فایل metadata در {base_path / 'metadata.json'} ذخیره شد.") 