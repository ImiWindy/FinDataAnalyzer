# مستندات FinDataAnalyzer

به مستندات FinDataAnalyzer خوش آمدید. این مستندات راهنمای کاملی برای استفاده از این پکیج تحلیل داده‌های مالی ارائه می‌دهد.

## معرفی

FinDataAnalyzer یک پکیج پایتون برای تحلیل داده‌های مالی است که امکاناتی مانند تحلیل آماری، تصویرسازی و پیش‌بینی روند قیمت را فراهم می‌کند. این پکیج برای کاربرانی که به دنبال تحلیل اتوماتیک داده‌های مالی هستند طراحی شده است.

## قابلیت‌ها

- **تحلیل داده‌های مالی:** آمار توصیفی، همبستگی، تشخیص روند و پرت‌داده‌ها
- **تصویرسازی:** نمودارهای قیمت، نمودار شمعی، نمودار همبستگی
- **پیش‌بینی:** پیش‌بینی قیمت با استفاده از الگوریتم‌های رگرسیون خطی و ARIMA
- **API:** رابط برنامه‌نویسی برای دسترسی به قابلیت‌های تحلیلی
- **داشبورد:** رابط کاربری تعاملی برای تحلیل و مصورسازی داده‌ها

## مستندات بخش‌ها

- [راهنمای شروع](usage.md): راهنمای استفاده از FinDataAnalyzer
- [API](api.md): مستندات API
- [راهنمای توسعه](development.md): راهنمای توسعه و مشارکت در پروژه

## نصب

برای نصب از pip استفاده کنید:

```bash
pip install findataanalyzer
```

یا مستقیماً از سورس:

```bash
git clone https://github.com/findataanalyzer/findataanalyzer.git
cd findataanalyzer
pip install -e .
```

## پیش‌نیازها

- Python 3.9+
- وابستگی‌های پکیج در فایل `requirements.txt`

## نمونه کد

```python
from findataanalyzer.core.analyzer import DataAnalyzer

# ایجاد یک آنالایزر
analyzer = DataAnalyzer()

# تحلیل داده‌های مالی
results = analyzer.analyze('path/to/data.csv')
print(results)
```

## حمایت و کمک

اگر سوالی دارید یا با مشکلی مواجه شده‌اید، می‌توانید:

- [ایشو جدید باز کنید](https://github.com/findataanalyzer/findataanalyzer/issues)
- به [صفحه ویکی پروژه](https://github.com/findataanalyzer/findataanalyzer/wiki) مراجعه کنید

## مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر به فایل [LICENSE](../LICENSE) مراجعه کنید. 