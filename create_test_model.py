"""
ایجاد یک مدل ساده برای تست
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# مسیر ذخیره مدل
model_dir = Path('models/test')
os.makedirs(model_dir, exist_ok=True)

# تنظیمات مدل
input_shape = (224, 224, 3)
num_classes = 5  # تعداد کلاس‌ها

# ایجاد مدل ساده CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# کامپایل مدل
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# نمایش خلاصه مدل
model.summary()

# ذخیره مدل
model_path = model_dir / 'model.h5'
model.save(str(model_path))

# ذخیره اطلاعات مدل
model_info = {
    "name": "TestModel",
    "architecture": "CNN",
    "input_shape": list(input_shape),
    "num_classes": num_classes,
    "date_created": time.ctime()
}

with open(model_dir / 'model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
    
print(f"\nمدل ساده تست با موفقیت در مسیر {model_path} ایجاد و ذخیره شد.") 