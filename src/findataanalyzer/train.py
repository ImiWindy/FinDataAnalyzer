"""اسکریپت آموزش مدل‌های تشخیص الگوی نمودار.

این اسکریپت امکان آموزش و ارزیابی مدل‌های تشخیص الگوهای تکنیکال در نمودارهای مالی را فراهم می‌کند.
"""

import os
import argparse
import logging
import json
import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

from findataanalyzer.utils.config import ConfigManager
from findataanalyzer.utils.logger import LogManager
from findataanalyzer.utils.data_manager import DataManager
from findataanalyzer.utils.gpu_utils import GPUManager
from findataanalyzer.utils.experiment_tracker import ExperimentTracker
from findataanalyzer.image_analysis.processors.image_processor import StandardProcessor
from findataanalyzer.image_analysis.database.image_database import GPUImageProcessor
from findataanalyzer.models.pattern_recognition.chart_pattern_model import ChartPatternModel, ChartPatternResNetModel


def setup_environment(config_path: str) -> Dict[str, Any]:
    """
    تنظیم محیط اجرا و بارگذاری پیکربندی.
    
    Args:
        config_path: مسیر فایل پیکربندی
    
    Returns:
        دیکشنری پیکربندی
    """
    # بارگذاری تنظیمات
    config_manager = ConfigManager(config_path)
    config = config_manager.get_all()
    
    # تنظیم لاگ
    log_manager = LogManager(config["logging"])
    log_manager.initialize()
    logger = log_manager.get_logger(__name__)
    logger.info("شروع فرآیند آموزش مدل...")
    
    # تنظیم GPU
    gpu_manager = GPUManager(config["gpu"])
    gpu_manager.initialize()
    
    return config


def load_data(config: Dict[str, Any], data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    بارگذاری داده‌های تصویر نمودار برای آموزش.
    
    Args:
        config: دیکشنری پیکربندی
        data_path: مسیر پوشه داده‌ها
    
    Returns:
        تاپل شامل تصاویر و برچسب‌ها
    """
    logger = logging.getLogger(__name__)
    logger.info(f"بارگذاری داده‌ها از مسیر {data_path}...")
    
    # مسیر داده‌ها
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise ValueError(f"مسیر داده‌ها یافت نشد: {data_path}")
    
    # بارگذاری فایل توضیحات داده‌ها (metadata) که شامل برچسب‌های تصاویر است
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"فایل metadata.json در مسیر {data_path} یافت نشد")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # دریافت تصاویر و برچسب‌ها
    images = []
    labels = []
    class_names = []
    
    if "classes" in metadata:
        class_names = metadata["classes"]
        logger.info(f"کلاس‌های شناسایی شده: {len(class_names)}")
    
    # پردازشگر تصویر برای پیش‌پردازش
    image_processor = GPUImageProcessor()
    
    # بارگذاری تصاویر و برچسب‌ها
    for item in metadata["images"]:
        image_path = data_dir / item["filename"]
        if not image_path.exists():
            logger.warning(f"تصویر {image_path} یافت نشد")
            continue
        
        try:
            # بارگذاری تصویر
            img = tf.io.read_file(str(image_path))
            img = tf.image.decode_image(img, channels=3)
            
            # تغییر اندازه تصویر
            img = tf.image.resize(img, (224, 224))
            
            # نرمال‌سازی
            img = img / 255.0
            
            images.append(img.numpy())
            labels.append(item["label"])
        except Exception as e:
            logger.error(f"خطا در بارگذاری تصویر {image_path}: {e}")
    
    logger.info(f"تعداد {len(images)} تصویر برای آموزش بارگذاری شد")
    
    # تبدیل به آرایه‌های NumPy
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, class_names


def prepare_data(X: np.ndarray, y: np.ndarray, val_split: float = 0.2) -> Dict[str, Any]:
    """
    آماده‌سازی داده‌ها برای آموزش و اعتبارسنجی.
    
    Args:
        X: آرایه تصاویر
        y: آرایه برچسب‌ها
        val_split: نسبت داده اعتبارسنجی
    
    Returns:
        دیکشنری دیتاست‌های آماده شده
    """
    logger = logging.getLogger(__name__)
    logger.info("آماده‌سازی داده‌ها برای آموزش...")
    
    # مخلوط کردن داده‌ها
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # تقسیم به مجموعه‌های آموزش و اعتبارسنجی
    val_size = int(len(X) * val_split)
    X_train, X_val = X[val_size:], X[:val_size]
    y_train, y_val = y[val_size:], y[:val_size]
    
    logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
    logger.info(f"تعداد نمونه‌های اعتبارسنجی: {len(X_val)}")
    
    # ایجاد دیتاست‌های TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # دسته‌بندی و پیش‌بارگیری داده‌ها
    batch_size = 32
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }


def create_model(config: Dict[str, Any], num_classes: int) -> ChartPatternModel:
    """
    ایجاد مدل تشخیص الگو.
    
    Args:
        config: دیکشنری پیکربندی
        num_classes: تعداد کلاس‌ها
        
    Returns:
        مدل ایجاد شده
    """
    logger = logging.getLogger(__name__)
    model_config = config.get("model", {})
    
    # اضافه کردن تعداد کلاس‌ها به پیکربندی
    model_config["num_classes"] = num_classes
    
    # انتخاب نوع مدل
    model_type = model_config.get("type", "default")
    logger.info(f"ایجاد مدل از نوع {model_type}...")
    
    if model_type == "resnet":
        model = ChartPatternResNetModel(model_config)
    else:
        model = ChartPatternModel(model_config)
        
    # ساخت مدل
    model.build_model()
    
    return model


def train_model(model: ChartPatternModel, dataset: Dict[str, Any], 
               config: Dict[str, Any], experiment_tracker: ExperimentTracker) -> Dict[str, Any]:
    """
    آموزش مدل.
    
    Args:
        model: مدل تشخیص الگو
        dataset: دیکشنری دیتاست‌ها
        config: دیکشنری پیکربندی
        experiment_tracker: ردیاب آزمایش
    
    Returns:
        تاریخچه آموزش
    """
    logger = logging.getLogger(__name__)
    logger.info("شروع آموزش مدل...")
    
    # تنظیمات آموزش
    train_config = config.get("training", {})
    epochs = train_config.get("epochs", 50)
    
    # ایجاد تابع‌های فراخوانی (callbacks)
    callbacks_list = model._get_default_callbacks()
    
    # افزودن تابع فراخوانی ردیاب آزمایش
    class ExperimentTrackerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                for metric_name, value in logs.items():
                    experiment_tracker.log_metric(metric_name, value, epoch)
    
    callbacks_list.append(ExperimentTrackerCallback())
    
    # شروع آموزش
    experiment_tracker.start_experiment(
        name="chart_pattern_detection",
        params={"model_config": config.get("model", {}),
                "train_config": train_config}
    )
    
    # آموزش مدل
    history = model.train(
        train_dataset=dataset["train_dataset"],
        validation_dataset=dataset["val_dataset"],
        class_names=[str(i) for i in range(model.config["num_classes"])],
        callbacks_list=callbacks_list
    )
    
    # پایان آزمایش
    experiment_tracker.end_experiment()
    
    return history


def evaluate_model(model: ChartPatternModel, dataset: Dict[str, Any]) -> Dict[str, float]:
    """
    ارزیابی مدل.
    
    Args:
        model: مدل آموزش دیده
        dataset: دیکشنری دیتاست‌ها
    
    Returns:
        دیکشنری معیارهای ارزیابی
    """
    logger = logging.getLogger(__name__)
    logger.info("ارزیابی مدل...")
    
    # ارزیابی مدل
    metrics = model.evaluate(dataset["val_dataset"])
    
    logger.info(f"نتایج ارزیابی: {metrics}")
    
    return metrics


def save_model(model: ChartPatternModel, output_dir: str, metrics: Dict[str, float]) -> str:
    """
    ذخیره مدل آموزش دیده.
    
    Args:
        model: مدل آموزش دیده
        output_dir: مسیر خروجی
        metrics: معیارهای ارزیابی
    
    Returns:
        مسیر ذخیره مدل
    """
    logger = logging.getLogger(__name__)
    
    # ایجاد مسیر خروجی
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # نام فایل با زمان
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_path / f"chart_pattern_model_{timestamp}"
    
    # ذخیره مدل
    model.save(str(model_path))
    
    # ذخیره معیارهای ارزیابی
    metrics_path = output_path / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"مدل با موفقیت در {model_path} ذخیره شد")
    logger.info(f"معیارهای ارزیابی در {metrics_path} ذخیره شد")
    
    return str(model_path)


def main():
    """تابع اصلی."""
    parser = argparse.ArgumentParser(description="آموزش مدل تشخیص الگوی نمودار")
    parser.add_argument('--config', type=str, default='configs/settings.yaml',
                      help='مسیر فایل پیکربندی')
    parser.add_argument('--data', type=str, required=True,
                      help='مسیر پوشه داده‌ها')
    parser.add_argument('--output', type=str, default='models/trained',
                      help='مسیر خروجی برای ذخیره مدل')
    args = parser.parse_args()
    
    try:
        # تنظیم محیط
        config = setup_environment(args.config)
        
        # ردیاب آزمایش
        experiment_tracker = ExperimentTracker(config["experiment_tracking"])
        
        # بارگذاری داده‌ها
        X, y, class_names = load_data(config, args.data)
        
        # آماده‌سازی داده‌ها
        dataset = prepare_data(X, y)
        
        # ایجاد مدل
        model = create_model(config, len(class_names) if class_names else len(np.unique(y)))
        
        # آموزش مدل
        history = train_model(model, dataset, config, experiment_tracker)
        
        # ارزیابی مدل
        metrics = evaluate_model(model, dataset)
        
        # ذخیره مدل
        model_path = save_model(model, args.output, metrics)
        
        logging.info(f"فرآیند آموزش با موفقیت تکمیل شد. مدل در {model_path} ذخیره شد.")
        
    except Exception as e:
        logging.error(f"خطا در فرآیند آموزش: {e}")
        raise


if __name__ == "__main__":
    main() 