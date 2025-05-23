"""ماژول تشخیص الگوهای نمودار با استفاده از یادگیری عمیق.

این ماژول مدل‌های تشخیص الگوهای تکنیکال در نمودارهای مالی را ارائه می‌دهد.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import json


class ChartPatternModel:
    """مدل تشخیص الگوهای نمودار مالی با استفاده از شبکه عصبی کانولوشنی."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """مقداردهی اولیه مدل.
        
        Args:
            config: پیکربندی مدل (اختیاری)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.model = None
        self.history = None
        self.class_names = None
        
        # بررسی پشتیبانی GPU
        self.gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
        if self.gpu_available:
            self.logger.info("مدل با پشتیبانی GPU اجرا خواهد شد.")
        else:
            self.logger.info("GPU در دسترس نیست، استفاده از CPU برای مدل.")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """پیکربندی پیش‌فرض مدل.
        
        Returns:
            دیکشنری پیکربندی پیش‌فرض
        """
        return {
            'input_shape': (224, 224, 1),  # ابعاد تصاویر ورودی (ارتفاع، عرض، کانال)
            'classes': 10,                 # تعداد کلاس‌ها (تعداد الگوهای مختلف)
            'learning_rate': 0.001,        # نرخ یادگیری
            'batch_size': 32,              # اندازه دسته
            'epochs': 50,                  # تعداد دوره‌ها
            'dropout_rate': 0.5,           # نرخ dropout
            'early_stopping_patience': 10, # صبر توقف زودهنگام
            'conv_layers': [               # لایه‌های کانولوشنی
                {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 256, 'kernel_size': 3, 'activation': 'relu'}
            ],
            'dense_layers': [              # لایه‌های متصل
                {'units': 512, 'activation': 'relu'},
                {'units': 256, 'activation': 'relu'}
            ],
            'checkpoints_dir': 'models/pattern_recognition'  # مسیر ذخیره چک‌پوینت‌ها
        }
    
    def build_model(self) -> tf.keras.Model:
        """ساخت معماری مدل.
        
        Returns:
            مدل TensorFlow
        """
        # پارامترهای معماری
        input_shape = self.config['input_shape']
        num_classes = self.config['classes']
        dropout_rate = self.config['dropout_rate']
        conv_layers = self.config['conv_layers']
        dense_layers = self.config['dense_layers']
        
        # ورودی مدل
        inputs = layers.Input(shape=input_shape)
        
        # لایه‌های کانولوشنی
        x = inputs
        for i, layer_config in enumerate(conv_layers):
            x = layers.Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config['activation'],
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f'pool_{i+1}')(x)
        
        # تغییر شکل به بردار
        x = layers.Flatten(name='flatten')(x)
        
        # لایه‌های متصل
        for i, layer_config in enumerate(dense_layers):
            x = layers.Dense(
                units=layer_config['units'],
                activation=layer_config['activation'],
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # لایه خروجی
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # ساخت مدل
        model = models.Model(inputs=inputs, outputs=outputs, name='chart_pattern_model')
        
        # کامپایل مدل
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # خلاصه مدل
        model.summary()
        
        self.model = model
        return model
    
    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, 
             class_names: List[str], callbacks_list: Optional[List[tf.keras.callbacks.Callback]] = None) -> Dict[str, List[float]]:
        """آموزش مدل.
        
        Args:
            train_dataset: دیتاست آموزش
            validation_dataset: دیتاست اعتبارسنجی
            class_names: نام کلاس‌ها
            callbacks_list: لیست callback‌ها (اختیاری)
            
        Returns:
            تاریخچه آموزش
        """
        if self.model is None:
            self.build_model()
        
        # تنظیم نام کلاس‌ها
        self.class_names = class_names
        
        # ایجاد مسیر ذخیره چک‌پوینت‌ها
        checkpoint_dir = self.config['checkpoints_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # آماده‌سازی callback‌ها
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        # آموزش مدل
        self.logger.info("شروع آموزش مدل...")
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
        self.logger.info("آموزش مدل به پایان رسید.")
        
        return self.history
    
    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """ایجاد callback‌های پیش‌فرض.
        
        Returns:
            لیست callback‌ها
        """
        checkpoint_path = os.path.join(
            self.config['checkpoints_dir'],
            'chart_pattern_model_checkpoint.h5'
        )
        
        return [
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config['checkpoints_dir'], 'logs'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """ارزیابی مدل.
        
        Args:
            test_dataset: دیتاست تست
            
        Returns:
            نتایج ارزیابی
        """
        if self.model is None:
            raise ValueError("مدل هنوز ساخته نشده است. ابتدا از متد build_model استفاده کنید.")
        
        self.logger.info("ارزیابی مدل...")
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # ذخیره نتایج
        metrics = {metric: value for metric, value in zip(self.model.metrics_names, results)}
        
        self.logger.info(f"ارزیابی مدل: {metrics}")
        return metrics
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """پیش‌بینی کلاس تصویر.
        
        Args:
            image: تصویر ورودی (باید با ابعاد مدل سازگار باشد)
            
        Returns:
            دیکشنری احتمالات کلاس‌ها
        """
        if self.model is None:
            raise ValueError("مدل هنوز ساخته نشده است. ابتدا از متد build_model استفاده کنید.")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("ورودی باید یک آرایه numpy باشد.")
        
        # اطمینان از ابعاد درست ورودی
        expected_shape = self.config['input_shape']
        if len(image.shape) == 3:  # تصویر منفرد
            image = np.expand_dims(image, axis=0)  # اضافه کردن بعد دسته
        
        if image.shape[1:] != expected_shape:
            raise ValueError(f"شکل تصویر ورودی {image.shape[1:]} با شکل مورد انتظار مدل {expected_shape} مطابقت ندارد.")
        
        # پیش‌بینی
        predictions = self.model.predict(image)
        
        # تبدیل به دیکشنری
        if self.class_names:
            result = {class_name: float(prob) for class_name, prob in zip(self.class_names, predictions[0])}
        else:
            result = {f"class_{i}": float(prob) for i, prob in enumerate(predictions[0])}
        
        return result
    
    def save(self, path: str) -> None:
        """ذخیره مدل.
        
        Args:
            path: مسیر ذخیره‌سازی
        """
        if self.model is None:
            raise ValueError("مدل هنوز ساخته نشده است. ابتدا از متد build_model استفاده کنید.")
        
        # ایجاد دایرکتوری اگر وجود ندارد
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # ذخیره مدل
        self.model.save(path)
        
        # ذخیره تنظیمات و کلاس‌ها
        config_path = f"{os.path.splitext(path)[0]}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'class_names': self.class_names,
                'history': self.history
            }, f)
        
        self.logger.info(f"مدل با موفقیت در {path} ذخیره شد.")
    
    def load(self, path: str) -> None:
        """بارگذاری مدل.
        
        Args:
            path: مسیر بارگذاری
        """
        # بررسی وجود فایل
        if not os.path.exists(path):
            raise ValueError(f"فایل مدل {path} وجود ندارد.")
        
        # بارگذاری مدل
        self.model = models.load_model(path)
        
        # بارگذاری تنظیمات و کلاس‌ها
        config_path = f"{os.path.splitext(path)[0]}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', self.config)
                self.class_names = data.get('class_names')
                self.history = data.get('history')
        
        self.logger.info(f"مدل با موفقیت از {path} بارگذاری شد.")


class ChartPatternResNetModel(ChartPatternModel):
    """مدل تشخیص الگوهای نمودار مالی با استفاده از ResNet."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """پیکربندی پیش‌فرض مدل.
        
        Returns:
            دیکشنری پیکربندی پیش‌فرض
        """
        config = super()._get_default_config()
        config.update({
            'blocks': [                    # بلوک‌های ResNet
                {'filters': 64, 'kernel_size': 3, 'blocks': 2},
                {'filters': 128, 'kernel_size': 3, 'blocks': 2},
                {'filters': 256, 'kernel_size': 3, 'blocks': 3},
                {'filters': 512, 'kernel_size': 3, 'blocks': 3}
            ],
            'dense_layers': [              # لایه‌های متصل
                {'units': 1024, 'activation': 'relu'},
                {'units': 512, 'activation': 'relu'}
            ]
        })
        return config
    
    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: int, downsample: bool = False) -> tf.Tensor:
        """بلوک باقیمانده (Residual Block).
        
        Args:
            x: تنسور ورودی
            filters: تعداد فیلترها
            kernel_size: اندازه کرنل
            downsample: کاهش ابعاد
            
        Returns:
            تنسور خروجی
        """
        identity = x
        
        # لایه اول
        stride = 2 if downsample else 1
        y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        # لایه دوم
        y = layers.Conv2D(filters, kernel_size, padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        # اتصال میانبر
        if downsample or x.shape[-1] != filters:
            identity = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
            identity = layers.BatchNormalization()(identity)
        
        # جمع
        output = layers.Add()([identity, y])
        output = layers.ReLU()(output)
        
        return output
    
    def build_model(self) -> tf.keras.Model:
        """ساخت معماری مدل ResNet.
        
        Returns:
            مدل TensorFlow
        """
        # پارامترهای معماری
        input_shape = self.config['input_shape']
        num_classes = self.config['classes']
        dropout_rate = self.config['dropout_rate']
        blocks = self.config['blocks']
        dense_layers = self.config['dense_layers']
        
        # ورودی مدل
        inputs = layers.Input(shape=input_shape)
        
        # لایه ورودی
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        # بلوک‌های ResNet
        for i, block_config in enumerate(blocks):
            filters = block_config['filters']
            kernel_size = block_config['kernel_size']
            num_blocks = block_config['blocks']
            
            # اولین بلوک هر دسته با کاهش ابعاد (به جز اولین دسته)
            x = self._residual_block(x, filters, kernel_size, downsample=(i > 0))
            
            # بلوک‌های اضافی بدون کاهش ابعاد
            for _ in range(1, num_blocks):
                x = self._residual_block(x, filters, kernel_size)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # لایه‌های متصل
        for layer_config in dense_layers:
            x = layers.Dense(layer_config['units'], activation=layer_config['activation'])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # لایه خروجی
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # ساخت مدل
        model = models.Model(inputs=inputs, outputs=outputs, name='chart_pattern_resnet_model')
        
        # کامپایل مدل
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # خلاصه مدل
        model.summary()
        
        self.model = model
        return model 