"""ماژول ردیابی آزمایش‌ها و نتایج.

این ماژول مسئول ثبت و مدیریت آزمایش‌ها، متریک‌ها و چک‌پوینت‌های مدل است.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    """مدیریت آزمایش‌ها و ردیابی نتایج."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه ردیاب آزمایش.
        
        Args:
            config: تنظیمات ردیابی آزمایش
        """
        self.config = config
        self.experiment_dir = Path(config['experiments_dir'])
        self.metrics_dir = Path(config['metrics_dir'])
        self.checkpoints_dir = Path(config['checkpoints_dir'])
        self.tensorboard_dir = Path(config['tensorboard_dir'])
        
        # ایجاد دایرکتوری‌های مورد نیاز
        self._create_directories()
        
        # تنظیم TensorBoard
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # اطلاعات آزمایش فعلی
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.start_time: Optional[float] = None
    
    def _create_directories(self) -> None:
        """ایجاد دایرکتوری‌های مورد نیاز."""
        for directory in [self.experiment_dir, self.metrics_dir, 
                         self.checkpoints_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def start_experiment(self, name: str, params: Dict[str, Any]) -> None:
        """
        شروع یک آزمایش جدید.
        
        Args:
            name: نام آزمایش
            params: پارامترهای آزمایش
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'params': params,
            'start_time': timestamp,
            'metrics': {},
            'checkpoints': []
        }
        
        self.start_time = time.time()
        
        # ذخیره اطلاعات آزمایش
        experiment_file = self.experiment_dir / f"{experiment_id}.json"
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_experiment, f, indent=4, ensure_ascii=False)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        ثبت یک متریک.
        
        Args:
            name: نام متریک
            value: مقدار متریک
            step: شماره گام (اختیاری)
        """
        if not self.current_experiment:
            raise RuntimeError("هیچ آزمایش فعالی وجود ندارد")
        
        # ثبت در TensorBoard
        self.writer.add_scalar(name, value, step)
        
        # ذخیره در دیکشنری متریک‌ها
        if name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][name] = []
        self.current_experiment['metrics'][name].append({
            'value': value,
            'step': step,
            'timestamp': time.time()
        })
        
        # به‌روزرسانی فایل آزمایش
        self._update_experiment_file()
    
    def save_checkpoint(self, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float]) -> None:
        """
        ذخیره چک‌پوینت مدل.
        
        Args:
            model: مدل
            optimizer: بهینه‌ساز
            epoch: شماره دوره
            metrics: متریک‌های دوره
        """
        if not self.current_experiment:
            raise RuntimeError("هیچ آزمایش فعالی وجود ندارد")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # ذخیره چک‌پوینت
        checkpoint_file = self.checkpoints_dir / f"{self.current_experiment['id']}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # ثبت در اطلاعات آزمایش
        self.current_experiment['checkpoints'].append({
            'epoch': epoch,
            'file': str(checkpoint_file),
            'metrics': metrics
        })
        
        # به‌روزرسانی فایل آزمایش
        self._update_experiment_file()
    
    def load_checkpoint(self, experiment_id: str, epoch: int) -> Dict[str, Any]:
        """
        بارگذاری چک‌پوینت.
        
        Args:
            experiment_id: شناسه آزمایش
            epoch: شماره دوره
            
        Returns:
            اطلاعات چک‌پوینت
        """
        checkpoint_file = self.checkpoints_dir / f"{experiment_id}_epoch_{epoch}.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"چک‌پوینت {checkpoint_file} یافت نشد")
        
        return torch.load(checkpoint_file)
    
    def end_experiment(self) -> None:
        """پایان آزمایش فعلی."""
        if not self.current_experiment:
            raise RuntimeError("هیچ آزمایش فعالی وجود ندارد")
        
        duration = time.time() - self.start_time
        self.current_experiment['duration'] = duration
        self.current_experiment['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # به‌روزرسانی فایل آزمایش
        self._update_experiment_file()
        
        # بستن TensorBoard
        self.writer.close()
        
        self.current_experiment = None
        self.start_time = None
    
    def _update_experiment_file(self) -> None:
        """به‌روزرسانی فایل اطلاعات آزمایش."""
        if not self.current_experiment:
            return
        
        experiment_file = self.experiment_dir / f"{self.current_experiment['id']}.json"
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_experiment, f, indent=4, ensure_ascii=False)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        دریافت خلاصه اطلاعات یک آزمایش.
        
        Args:
            experiment_id: شناسه آزمایش
            
        Returns:
            خلاصه اطلاعات آزمایش
        """
        experiment_file = self.experiment_dir / f"{experiment_id}.json"
        if not experiment_file.exists():
            raise FileNotFoundError(f"آزمایش {experiment_id} یافت نشد")
        
        with open(experiment_file, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)
        
        return {
            'id': experiment_data['id'],
            'name': experiment_data['name'],
            'duration': experiment_data.get('duration'),
            'start_time': experiment_data['start_time'],
            'end_time': experiment_data.get('end_time'),
            'final_metrics': {
                name: values[-1]['value'] 
                for name, values in experiment_data['metrics'].items()
            } if 'metrics' in experiment_data else {}
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست تمام آزمایش‌ها.
        
        Returns:
            لیست خلاصه اطلاعات آزمایش‌ها
        """
        experiments = []
        for experiment_file in self.experiment_dir.glob("*.json"):
            with open(experiment_file, 'r', encoding='utf-8') as f:
                experiment_data = json.load(f)
                experiments.append(self.get_experiment_summary(experiment_data['id']))
        
        return sorted(experiments, key=lambda x: x['start_time'], reverse=True) 