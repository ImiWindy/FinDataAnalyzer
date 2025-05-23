"""کلاس Trainer برای آموزش و ارزیابی مدل‌های تشخیص الگو.

این ماژول کلاس‌هایی برای آموزش و ارزیابی مدل‌های تشخیص الگو در نمودارهای مالی ارائه می‌دهد.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

from findataanalyzer.image_analysis.models.base_model import BasePatternModel


class ModelTrainer:
    """کلاس Trainer برای آموزش و ارزیابی مدل‌های تشخیص الگو."""
    
    def __init__(self, model: BasePatternModel, config: Dict[str, Any]):
        """
        مقداردهی اولیه Trainer.
        
        Args:
            model: مدل برای آموزش
            config: تنظیمات آموزش شامل:
                - batch_size: اندازه دسته
                - num_epochs: تعداد دوره‌های آموزش
                - early_stopping_patience: صبر برای توقف زودهنگام
                - learning_rate: نرخ یادگیری
                - device: دستگاه محاسباتی (cuda/cpu)
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # تنظیمات آموزش
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # انتقال مدل به دستگاه محاسباتی
        self.model = self.model.to(self.device)
        
        # تاریخچه آموزش
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        آموزش مدل.
        
        Args:
            train_loader: DataLoader برای داده‌های آموزش
            val_loader: DataLoader برای داده‌های اعتبارسنجی (اختیاری)
            
        Returns:
            تاریخچه آموزش
        """
        self.logger.info("شروع آموزش مدل...")
        
        # متغیرهای توقف زودهنگام
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # آموزش
            train_metrics = self._train_epoch(train_loader)
            
            # اعتبارسنجی
            val_metrics = self._validate_epoch(val_loader) if val_loader else None
            
            # ثبت معیارها
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_accuracy'].append(val_metrics['val_accuracy'])
            
            # چاپ نتایج
            epoch_time = time.time() - start_time
            self._log_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
            
            # بررسی توقف زودهنگام
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                # ذخیره بهترین مدل
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"توقف زودهنگام در دوره {epoch + 1}")
                    break
            
            # ذخیره چک‌پوینت دوره
            self._save_checkpoint(epoch)
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        آموزش یک دوره.
        
        Args:
            train_loader: DataLoader برای داده‌های آموزش
            
        Returns:
            معیارهای آموزش
        """
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="آموزش") as pbar:
            for batch in pbar:
                # انتقال داده به دستگاه محاسباتی
                inputs, targets = [b.to(self.device) for b in batch]
                
                # آموزش یک دسته
                metrics = self.model.train_step((inputs, targets))
                
                # به‌روزرسانی معیارها
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                
                # به‌روزرسانی نوار پیشرفت
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'accuracy': f"{metrics['accuracy']:.4f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        اعتبارسنجی یک دوره.
        
        Args:
            val_loader: DataLoader برای داده‌های اعتبارسنجی
            
        Returns:
            معیارهای اعتبارسنجی
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with tqdm(val_loader, desc="اعتبارسنجی") as pbar:
                for batch in pbar:
                    # انتقال داده به دستگاه محاسباتی
                    inputs, targets = [b.to(self.device) for b in batch]
                    
                    # اعتبارسنجی یک دسته
                    metrics = self.model.validate_step((inputs, targets))
                    
                    # به‌روزرسانی معیارها
                    total_loss += metrics['val_loss']
                    total_accuracy += metrics['val_accuracy']
                    
                    # به‌روزرسانی نوار پیشرفت
                    pbar.set_postfix({
                        'val_loss': f"{metrics['val_loss']:.4f}",
                        'val_accuracy': f"{metrics['val_accuracy']:.4f}"
                    })
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Optional[Dict[str, float]], epoch_time: float) -> None:
        """
        ثبت نتایج دوره.
        
        Args:
            epoch: شماره دوره
            train_metrics: معیارهای آموزش
            val_metrics: معیارهای اعتبارسنجی
            epoch_time: زمان دوره
        """
        log_msg = f"دوره {epoch + 1}/{self.num_epochs} - "
        log_msg += f"زمان: {epoch_time:.2f} ثانیه - "
        log_msg += f"آموزش: loss={train_metrics['loss']:.4f}, accuracy={train_metrics['accuracy']:.4f}"
        
        if val_metrics:
            log_msg += f" - اعتبارسنجی: val_loss={val_metrics['val_loss']:.4f}, "
            log_msg += f"val_accuracy={val_metrics['val_accuracy']:.4f}"
        
        self.logger.info(log_msg)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        ذخیره چک‌پوینت.
        
        Args:
            epoch: شماره دوره
            is_best: آیا این بهترین مدل است؟
        """
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # ذخیره چک‌پوینت دوره
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        self.model.save(str(checkpoint_path))
        
        # ذخیره بهترین مدل
        if is_best:
            best_model_path = checkpoint_dir / "best_model.pt"
            self.model.save(str(best_model_path))
            
            # ذخیره تاریخچه
            history_path = checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        ارزیابی مدل روی داده‌های تست.
        
        Args:
            test_loader: DataLoader برای داده‌های تست
            
        Returns:
            معیارهای ارزیابی
        """
        self.logger.info("ارزیابی مدل...")
        
        # بارگذاری بهترین مدل
        best_model_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / "best_model.pt"
        if best_model_path.exists():
            self.model.load(str(best_model_path))
        
        # ارزیابی
        metrics = self._validate_epoch(test_loader)
        
        self.logger.info(f"نتایج ارزیابی: loss={metrics['val_loss']:.4f}, accuracy={metrics['val_accuracy']:.4f}")
        
        return metrics 