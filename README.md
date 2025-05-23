# FinDataAnalyzer

سیستم تحلیل نمودارهای بازار مالی با استفاده از هوش مصنوعی

## معرفی

FinDataAnalyzer یک سیستم تحلیل داده مالی است که با استفاده از الگوریتم‌های پردازش تصویر و یادگیری عمیق، الگوهای تکنیکال را در نمودارهای قیمت شناسایی می‌کند. این سیستم با TensorFlow پیاده‌سازی شده و قابلیت اجرا روی GPU را دارد.

## ویژگی‌ها

- تشخیص الگوهای تکنیکال در نمودارهای قیمت
- پردازش تصاویر نمودار با استفاده از GPU
- ذخیره‌سازی و مدیریت داده‌های تاریخی
- API برای یکپارچه‌سازی با سایر سیستم‌ها
- داشبورد مدیریتی برای مشاهده تحلیل‌ها

## نیازمندی‌ها

- Python 3.8 یا بالاتر
- TensorFlow 2.5 یا بالاتر (برای پشتیبانی GPU)
- FastAPI (برای API)
- SQLite (برای ذخیره‌سازی داده)
- سایر وابستگی‌ها در فایل requirements.txt

## نصب و راه‌اندازی

1. کلون کردن مخزن:

```bash
git clone https://github.com/yourusername/findataanalyzer.git
cd findataanalyzer
```

2. نصب وابستگی‌ها:

```bash
# نصب با پشتیبانی GPU (توصیه شده)
python src/findataanalyzer/scripts/setup_server.py --install-deps --gpu

# یا نصب بدون پشتیبانی GPU
python src/findataanalyzer/scripts/setup_server.py --install-deps
```

3. راه‌اندازی اولیه سیستم:

```bash
python src/findataanalyzer/scripts/setup_server.py --setup-dirs --setup-db
```

4. اجرای سرور API:

```bash
python -m findataanalyzer.api.main
```

## استفاده

### API

سرور API روی پورت 8000 راه‌اندازی می‌شود. می‌توانید با مراجعه به آدرس زیر به مستندات API دسترسی پیدا کنید:

```
http://localhost:8000/docs
```

### پردازش تصاویر نمودار

برای پردازش تصاویر نمودار و تشخیص الگوها:

```python
from findataanalyzer.image_analysis.processors.feature_extractor import FeatureExtractor

# ایجاد نمونه از استخراج‌کننده ویژگی
extractor = FeatureExtractor()

# تشخیص الگوهای نمودار
chart_patterns = extractor.detect_chart_patterns("path/to/chart/image.png")
print(chart_patterns)
```

### آموزش مدل جدید

برای آموزش یک مدل تشخیص الگو جدید:

```python
from findataanalyzer.image_analysis.models.cnn_lstm_model import CNNLSTMPatternModel
from findataanalyzer.image_analysis.models.trainer import ModelTrainer

# تنظیمات مدل
model_config = {
    "input_channels": 1,
    "lstm_hidden_size": 256,
    "num_classes": 10,
}

# ایجاد مدل
model = CNNLSTMPatternModel(model_config)

# تنظیمات آموزش
trainer_config = {
    "batch_size": 32,
    "num_epochs": 100,
    "device": "GPU"  # یا "CPU"
}

# ایجاد آموزش‌دهنده
trainer = ModelTrainer(model, trainer_config)

# آموزش مدل (نیاز به آماده‌سازی داده‌های آموزش)
trainer.train(train_loader, val_loader)
```

## مجوز

این پروژه تحت مجوز MIT منتشر شده است.
