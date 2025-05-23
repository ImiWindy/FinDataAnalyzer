"""ماژول مدیریت مدل‌ها.

این ماژول مسئول مدیریت مدل‌های یادگیری ماشین و عملیات مربوط به آن‌ها است.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime


class ModelManager:
    """مدیریت مدل‌ها."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه مدیر مدل.
        
        Args:
            config: تنظیمات مدیریت مدل
        """
        self.config = config
        self.models_dir = Path(config['models_dir'])
        self.checkpoints_dir = Path(config['checkpoints_dir'])
        
        # ایجاد دایرکتوری‌های مورد نیاز
        self._create_directories()
        
        # تنظیم دستگاه محاسباتی
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_directories(self) -> None:
        """ایجاد دایرکتوری‌های مورد نیاز."""
        for directory in [self.models_dir, self.checkpoints_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: nn.Module, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        ذخیره مدل.
        
        Args:
            model: مدل
            name: نام مدل
            metadata: اطلاعات تکمیلی (اختیاری)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # ذخیره مدل
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # ذخیره اطلاعات تکمیلی
        if metadata:
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    def load_model(self, model: nn.Module, name: str, timestamp: Optional[str] = None) -> nn.Module:
        """
        بارگذاری مدل.
        
        Args:
            model: مدل
            name: نام مدل
            timestamp: زمان ذخیره (اختیاری)
            
        Returns:
            مدل بارگذاری شده
        """
        if timestamp:
            model_dir = self.models_dir / f"{name}_{timestamp}"
        else:
            # یافتن آخرین مدل
            model_dirs = list(self.models_dir.glob(f"{name}_*"))
            if not model_dirs:
                raise FileNotFoundError(f"مدلی با نام {name} یافت نشد")
            model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"فایل مدل در {model_path} یافت نشد")
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def get_model_metadata(self, name: str, timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات تکمیلی مدل.
        
        Args:
            name: نام مدل
            timestamp: زمان ذخیره (اختیاری)
            
        Returns:
            اطلاعات تکمیلی مدل
        """
        if timestamp:
            model_dir = self.models_dir / f"{name}_{timestamp}"
        else:
            model_dirs = list(self.models_dir.glob(f"{name}_*"))
            if not model_dirs:
                return None
            model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        ذخیره چک‌پوینت.
        
        Args:
            model: مدل
            optimizer: بهینه‌ساز
            epoch: شماره دوره
            loss: مقدار تابع هزینه
            metadata: اطلاعات تکمیلی (اختیاری)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int) -> Tuple[nn.Module, torch.optim.Optimizer, int, float, Dict[str, Any]]:
        """
        بارگذاری چک‌پوینت.
        
        Args:
            model: مدل
            optimizer: بهینه‌ساز
            epoch: شماره دوره
            
        Returns:
            مدل، بهینه‌ساز، شماره دوره، مقدار تابع هزینه و اطلاعات تکمیلی
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"چک‌پوینت دوره {epoch} یافت نشد")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return (model, optimizer, checkpoint['epoch'], 
                checkpoint['loss'], checkpoint['metadata'])
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست مدل‌ها.
        
        Returns:
            لیست اطلاعات مدل‌ها
        """
        models = []
        for model_dir in self.models_dir.glob("*_*"):
            name, timestamp = model_dir.stem.split('_', 1)
            
            metadata = self.get_model_metadata(name, timestamp)
            
            models.append({
                'name': name,
                'timestamp': timestamp,
                'metadata': metadata,
                'path': str(model_dir)
            })
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست چک‌پوینت‌ها.
        
        Returns:
            لیست اطلاعات چک‌پوینت‌ها
        """
        checkpoints = []
        for checkpoint_path in self.checkpoints_dir.glob("checkpoint_epoch_*.pt"):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            checkpoints.append({
                'epoch': checkpoint['epoch'],
                'loss': checkpoint['loss'],
                'metadata': checkpoint['metadata'],
                'path': str(checkpoint_path)
            })
        
        return sorted(checkpoints, key=lambda x: x['epoch'])
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """
        ارزیابی مدل.
        
        Args:
            model: مدل
            data_loader: لودر داده
            criterion: تابع هزینه
            
        Returns:
            نتایج ارزیابی
        """
        model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        return {
            'loss': total_loss / len(data_loader),
            'mse': np.mean((predictions - targets) ** 2),
            'mae': np.mean(np.abs(predictions - targets)),
            'r2': 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        } 