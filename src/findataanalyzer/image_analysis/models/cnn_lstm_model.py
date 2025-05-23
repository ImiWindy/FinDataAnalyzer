"""مدل CNN-LSTM برای تشخیص الگوهای تکنیکال در نمودارهای مالی.

این ماژول یک مدل ترکیبی از شبکه‌های عصبی کانولوشنی و LSTM برای تشخیص الگوهای تکنیکال ارائه می‌دهد.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Dict, Any, List, Tuple
import logging
import numpy as np


class CNNLSTMPatternModel:
    """مدل CNN-LSTM برای تشخیص الگوهای تکنیکال."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدل.
        
        Args:
            config: دیکشنری تنظیمات مدل شامل:
                - input_channels: تعداد کانال‌های ورودی (معمولاً 1 برای تصاویر سطح خاکستری)
                - cnn_layers: لیست تنظیمات لایه‌های CNN
                - lstm_hidden_size: اندازه لایه پنهان LSTM
                - lstm_num_layers: تعداد لایه‌های LSTM
                - num_classes: تعداد کلاس‌های خروجی
                - dropout_rate: نرخ Dropout
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # تنظیمات مدل
        self.input_channels = config.get('input_channels', 1)
        self.cnn_layers = config.get('cnn_layers', [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ])
        self.lstm_hidden_size = config.get('lstm_hidden_size', 256)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # ایجاد مدل
        self._build_model()
        
        # تاریخچه آموزش
        self.is_trained = False
        self.training_history = []
    
    def _build_model(self):
        """ساخت مدل با استفاده از TensorFlow/Keras."""
        # تعریف ورودی
        input_shape = (224, 224, self.input_channels)  # TensorFlow از فرمت (H, W, C) استفاده می‌کند
        inputs = layers.Input(shape=input_shape)
        
        # بخش CNN
        x = inputs
        for i, layer_config in enumerate(self.cnn_layers):
            x = layers.Conv2D(
                filters=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                strides=layer_config['stride'],
                padding='same' if layer_config['padding'] else 'valid',
                activation='relu'
            )(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        # تغییر شکل برای LSTM
        # ابتدا به شکل (batch, time_steps, features) تبدیل می‌کنیم
        _, h, w, c = x.shape
        x = layers.Reshape((h, w * c))(x)
        
        # بخش LSTM
        for i in range(self.lstm_num_layers):
            return_sequences = i < self.lstm_num_layers - 1  # فقط در آخرین لایه return_sequences=False
            x = layers.LSTM(
                units=self.lstm_hidden_size,
                return_sequences=return_sequences,
                dropout=self.dropout_rate if i < self.lstm_num_layers - 1 else 0
            )(x)
        
        # بخش طبقه‌بندی
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # ایجاد مدل
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # کامپایل مدل
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # نمایش خلاصه مدل
        self.model.summary()
    
    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        انجام یک مرحله آموزش.
        
        Args:
            batch: تاپل (ورودی‌ها، برچسب‌ها)
            
        Returns:
            دیکشنری معیارهای آموزش
        """
        inputs, targets = batch
        
        # استفاده از tape برای ثبت عملیات محاسبه گرادیان
        with tf.GradientTape() as tape:
            # پیش‌بینی‌ها
            predictions = self.model(inputs, training=True)
            # محاسبه ضرر
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions)
            loss = tf.reduce_mean(loss)
        
        # محاسبه گرادیان‌ها
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # به‌روزرسانی وزن‌ها
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # محاسبه دقت
        predictions = tf.argmax(predictions, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(targets, tf.int64)), tf.float32))
        
        return {
            'loss': loss.numpy().item(),
            'accuracy': accuracy.numpy().item()
        }
    
    def validate_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        انجام یک مرحله اعتبارسنجی.
        
        Args:
            batch: تاپل (ورودی‌ها، برچسب‌ها)
            
        Returns:
            دیکشنری معیارهای اعتبارسنجی
        """
        inputs, targets = batch
        
        # پیش‌بینی‌ها
        predictions = self.model(inputs, training=False)
        # محاسبه ضرر
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions)
        loss = tf.reduce_mean(loss)
        
        # محاسبه دقت
        predictions = tf.argmax(predictions, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(targets, tf.int64)), tf.float32))
        
        return {
            'val_loss': loss.numpy().item(),
            'val_accuracy': accuracy.numpy().item()
        }
    
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """پیش‌بینی با استفاده از مدل."""
        return self.model(x, training=False)
    
    def save(self, path: str) -> None:
        """ذخیره مدل در مسیر مشخص شده."""
        try:
            # ذخیره مدل TensorFlow
            self.model.save(path)
            
            # ذخیره تنظیمات و تاریخچه آموزش
            import json
            import os
            
            # ذخیره تنظیمات
            config_path = os.path.join(os.path.dirname(path), "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "config": self.config,
                    "is_trained": self.is_trained,
                    "training_history": self.training_history
                }, f)
            
            self.logger.info(f"مدل با موفقیت در {path} ذخیره شد")
        except Exception as e:
            self.logger.error(f"خطا در ذخیره مدل: {e}")
    
    def load(self, path: str) -> None:
        """بارگذاری مدل از مسیر مشخص شده."""
        try:
            # بارگذاری مدل TensorFlow
            self.model = tf.keras.models.load_model(path)
            
            # بارگذاری تنظیمات و تاریخچه آموزش
            import json
            import os
            
            # بارگذاری تنظیمات
            config_path = os.path.join(os.path.dirname(path), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = json.load(f)
                    self.config = data.get("config", self.config)
                    self.is_trained = data.get("is_trained", False)
                    self.training_history = data.get("training_history", [])
            
            self.logger.info(f"مدل با موفقیت از {path} بارگذاری شد")
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری مدل: {e}")
    
    def evaluate(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        """ارزیابی مدل روی داده‌های تست."""
        results = self.model.evaluate(x, y, verbose=0)
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
    
    def get_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات مدل."""
        return self.config
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """دریافت تاریخچه آموزش مدل."""
        return self.training_history 