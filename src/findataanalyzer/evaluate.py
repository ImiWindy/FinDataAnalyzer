"""اسکریپت ارزیابی مدل‌های تشخیص الگوی نمودار.

این اسکریپت امکان ارزیابی مدل‌های تشخیص الگوهای تکنیکال در نمودارهای مالی روی داده‌های جدید را فراهم می‌کند.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# مدیریت وابستگی‌های اختیاری
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("پکیج seaborn نصب نشده است. برخی از نمودارها با کیفیت کمتری نمایش داده خواهند شد.")

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("پکیج TSNE در sklearn نصب نشده است. تجسم TSNE انجام نخواهد شد.")


# تنظیم لاگر ساده برای اجرای مستقل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> tf.keras.Model:
    """
    بارگذاری مدل آموزش دیده.
    
    Args:
        model_path: مسیر مدل ذخیره شده
    
    Returns:
        مدل بارگذاری شده
    """
    logger.info(f"بارگذاری مدل از مسیر {model_path}...")
    
    # بررسی وجود مسیر مدل
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"مدل در مسیر {model_path} یافت نشد")
    
    # بارگذاری مدل
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("مدل با موفقیت بارگذاری شد")
        return model
    except Exception as e:
        logger.error(f"خطا در بارگذاری مدل: {e}")
        raise


def load_test_data(data_path: str, image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    بارگذاری داده‌های تصویر نمودار برای تست.
    
    Args:
        data_path: مسیر پوشه داده‌های تست
        image_size: اندازه تصاویر برای تغییر ابعاد
    
    Returns:
        تاپل شامل تصاویر، برچسب‌ها، نام کلاس‌ها و مسیرهای تصاویر
    """
    logger.info(f"بارگذاری داده‌های تست از مسیر {data_path}...")
    
    # مسیر داده‌ها
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise ValueError(f"مسیر داده‌های تست یافت نشد: {data_path}")
    
    # بارگذاری فایل توضیحات داده‌ها (metadata) که شامل برچسب‌های تصاویر است
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"فایل metadata.json در مسیر {data_path} یافت نشد")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # دریافت تصاویر و برچسب‌ها
    images = []
    labels = []
    image_paths = []
    class_names = []
    
    if "classes" in metadata:
        class_names = metadata["classes"]
        logger.info(f"کلاس‌های شناسایی شده: {len(class_names)}")
    
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
            img = tf.image.resize(img, image_size)
            
            # نرمال‌سازی
            img = img / 255.0
            
            images.append(img.numpy())
            labels.append(item["label"])
            image_paths.append(str(image_path))
        except Exception as e:
            logger.error(f"خطا در بارگذاری تصویر {image_path}: {e}")
    
    logger.info(f"تعداد {len(images)} تصویر برای تست بارگذاری شد")
    
    # تبدیل به آرایه‌های NumPy
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, class_names, image_paths


def evaluate_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, 
                  class_names: List[str]) -> Dict[str, Any]:
    """
    ارزیابی مدل روی داده‌های تست.
    
    Args:
        model: مدل آموزش دیده
        X: آرایه تصاویر تست
        y: آرایه برچسب‌های تست
        class_names: نام کلاس‌ها
    
    Returns:
        دیکشنری معیارهای ارزیابی
    """
    logger.info("ارزیابی مدل روی داده‌های تست...")
    
    # تبدیل داده‌ها به دیتاست TensorFlow
    test_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # ارزیابی مدل
    loss, accuracy = model.evaluate(test_dataset)
    logger.info(f"دقت ارزیابی مدل: {accuracy:.4f}")
    
    # پیش‌بینی
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    # ایجاد گزارش طبقه‌بندی
    if class_names and len(class_names) == len(np.unique(y)):
        target_names = class_names
    else:
        target_names = [str(i) for i in range(len(np.unique(y)))]
    
    classification_rep = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": classification_rep,
        "confusion_matrix": conf_matrix.tolist(),
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_probs": y_pred_probs.tolist()
    }
    
    return results


def visualize_results(results: Dict[str, Any], class_names: List[str], 
                     X: np.ndarray, y: np.ndarray, output_dir: str) -> None:
    """
    تجسم نتایج ارزیابی.
    
    Args:
        results: نتایج ارزیابی
        class_names: نام کلاس‌ها
        X: داده‌های تصویر
        y: برچسب‌های واقعی
        output_dir: مسیر خروجی برای ذخیره تصاویر
    """
    logger.info("ایجاد تجسم‌های نتایج ارزیابی...")
    
    # ایجاد مسیر خروجی
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ماتریس درهم‌ریختگی (Confusion Matrix)
    plt.figure(figsize=(10, 8))
    conf_matrix = np.array(results["confusion_matrix"])
    
    if HAS_SEABORN:
        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('پیش‌بینی شده')
        ax.set_ylabel('واقعی')
    else:
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.xlabel('پیش‌بینی شده')
        plt.ylabel('واقعی')
    
    plt.title('ماتریس درهم‌ریختگی')
    plt.tight_layout()
    plt.savefig(str(output_path / "confusion_matrix.png"))
    plt.close()
    
    # نمودار دقت به ازای هر کلاس
    plt.figure(figsize=(12, 6))
    class_accuracies = []
    
    for i, name in enumerate(class_names):
        if str(i) in results["classification_report"]:
            class_accuracies.append(results["classification_report"][str(i)]["precision"])
        elif name in results["classification_report"]:
            class_accuracies.append(results["classification_report"][name]["precision"])
        else:
            class_accuracies.append(0)
    
    plt.bar(class_names, class_accuracies)
    plt.xlabel('کلاس‌ها')
    plt.ylabel('دقت')
    plt.title('دقت به ازای هر کلاس')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(output_path / "class_precision.png"))
    plt.close()
    
    # اگر TSNE موجود باشد، تجسم داده‌ها
    if HAS_TSNE and len(X) > 10:
        logger.info("ایجاد تجسم TSNE از داده‌ها...")
        try:
            # گرفتن ویژگی‌ها از لایه قبل از آخر
            # این بخش بستگی به معماری مدل دارد
            if hasattr(results, "features"):
                features = results["features"]
            else:
                # ایجاد مدل برای استخراج ویژگی‌ها
                base_model = model
                # فرض می‌کنیم آخرین لایه dense است
                feature_model = tf.keras.Model(inputs=base_model.input, 
                                            outputs=base_model.layers[-2].output)
                features = feature_model.predict(X)
            
            # کاهش ابعاد با TSNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(features)
            
            # رسم نتایج
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.title('تجسم TSNE از ویژگی‌های استخراج شده')
            plt.xlabel('TSNE-1')
            plt.ylabel('TSNE-2')
            plt.savefig(str(output_path / "tsne_visualization.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"خطا در ایجاد تجسم TSNE: {e}")
    
    # نمایش نمونه‌های اشتباه طبقه‌بندی شده
    y_pred = results["y_pred"]
    misclassified_indices = np.where(np.array(y) != np.array(y_pred))[0]
    
    if len(misclassified_indices) > 0:
        n_samples = min(10, len(misclassified_indices))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(misclassified_indices[:n_samples]):
            axes[i].imshow(X[idx])
            axes[i].set_title(f'واقعی: {class_names[y[idx]]}\nپیش‌بینی: {class_names[y_pred[idx]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_path / "misclassified_samples.png"))
        plt.close()
    
    logger.info(f"تجسم‌ها با موفقیت در مسیر {output_dir} ذخیره شدند")


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    ذخیره نتایج ارزیابی.
    
    Args:
        results: نتایج ارزیابی
        output_dir: مسیر خروجی
    """
    
    # ایجاد مسیر خروجی
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # نام فایل با زمان
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_path / f"evaluation_results_{timestamp}.json"
    
    # تبدیل آرایه‌های NumPy به لیست برای ذخیره در JSON
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        else:
            serializable_results[k] = v
    
    # ذخیره نتایج
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # ایجاد فایل خلاصه نتایج
    summary_path = output_path / f"evaluation_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("خلاصه نتایج ارزیابی مدل تشخیص الگوی نمودار\n")
        f.write("="*50 + "\n\n")
        f.write(f"تاریخ ارزیابی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"دقت (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"دقت (Precision): {results['precision']:.4f}\n")
        f.write(f"فراخوانی (Recall): {results['recall']:.4f}\n")
        f.write(f"معیار F1: {results['f1_score']:.4f}\n\n")
        f.write("گزارش طبقه‌بندی:\n")
        for cls, metrics in results["classification_report"].items():
            if isinstance(metrics, dict):
                f.write(f"  کلاس {cls}:\n")
                f.write(f"    دقت (Precision): {metrics.get('precision', 0):.4f}\n")
                f.write(f"    فراخوانی (Recall): {metrics.get('recall', 0):.4f}\n")
                f.write(f"    معیار F1: {metrics.get('f1-score', 0):.4f}\n")
                f.write(f"    تعداد نمونه‌ها: {metrics.get('support', 0)}\n")
    
    logger.info(f"نتایج ارزیابی با موفقیت در {results_path} ذخیره شد")
    logger.info(f"خلاصه نتایج در {summary_path} ذخیره شد")


def main():
    """تابع اصلی."""
    parser = argparse.ArgumentParser(description="ارزیابی مدل تشخیص الگوی نمودار")
    parser.add_argument('--model', type=str, required=True,
                      help='مسیر مدل آموزش دیده')
    parser.add_argument('--data', type=str, required=True,
                      help='مسیر پوشه داده‌های تست')
    parser.add_argument('--output', type=str, default='results/evaluation',
                      help='مسیر خروجی برای ذخیره نتایج')
    parser.add_argument('--visualize', action='store_true',
                      help='تجسم نتایج ارزیابی')
    args = parser.parse_args()
    
    try:
        # بارگذاری مدل
        model = load_model(args.model)
        
        # بارگذاری داده‌های تست
        X, y, class_names, image_paths = load_test_data(args.data)
        
        # ارزیابی مدل
        results = evaluate_model(model, X, y, class_names)
        
        # ذخیره نتایج
        save_results(results, args.output)
        
        # تجسم نتایج (اختیاری)
        if args.visualize:
            visualize_results(results, class_names, X, y, args.output)
        
        logger.info(f"فرآیند ارزیابی با موفقیت تکمیل شد. نتایج در {args.output} ذخیره شدند.")
        
    except Exception as e:
        logger.error(f"خطا در فرآیند ارزیابی: {e}")
        raise


if __name__ == "__main__":
    main() 