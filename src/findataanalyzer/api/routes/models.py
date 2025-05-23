"""مسیرهای API برای مدیریت مدل‌ها و پیش‌بینی.

این ماژول APIهایی برای آموزش، ارزیابی و مدیریت مدل‌های یادگیری ماشین ارائه می‌دهد.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from pydantic import BaseModel

from findataanalyzer.utils.model_manager import ModelManager
from findataanalyzer.utils.config import ConfigManager
from findataanalyzer.image_analysis.models.cnn_lstm_model import CNNLSTMPatternModel
from findataanalyzer.image_analysis.models.trainer import ModelTrainer
from findataanalyzer.utils.data_manager import DataManager
from findataanalyzer.utils.experiment_tracker import ExperimentTracker

router = APIRouter(
    prefix="/models",
    tags=["models"],
)

# لود کردن پیکربندی
config_manager = ConfigManager("configs/settings.yaml")
config = config_manager.get_all()

# مدیرهای مورد نیاز
model_manager = ModelManager(config["model"])
data_manager = DataManager(config["data_pipeline"])
experiment_tracker = ExperimentTracker(config["experiment_tracking"])

# مدل‌های داده
class TrainingRequest(BaseModel):
    """مدل درخواست آموزش."""
    
    model_name: str
    model_type: str = "cnn_lstm"
    symbols: List[str]
    target_column: str
    feature_columns: Optional[List[str]] = None
    hyperparameters: Dict[str, Any] = {}
    experiment_name: str
    description: Optional[str] = None


class TrainingResponse(BaseModel):
    """مدل پاسخ آموزش."""
    
    success: bool
    message: str
    experiment_id: Optional[str] = None
    model_id: Optional[str] = None


class PredictionRequest(BaseModel):
    """مدل درخواست پیش‌بینی."""
    
    model_id: str
    image_url: Optional[str] = None
    data: Optional[Dict[str, List[float]]] = None


class PredictionResponse(BaseModel):
    """مدل پاسخ پیش‌بینی."""
    
    success: bool
    predictions: List[Dict[str, Any]]
    confidence: float
    message: str


# متغیر سراسری برای تعقیب آموزش‌های در حال اجرا
active_trainings: Dict[str, Dict[str, Any]] = {}


async def train_model_task(request: TrainingRequest, experiment_id: str):
    """
    تابع آموزش مدل در پس‌زمینه.
    
    Args:
        request: درخواست آموزش
        experiment_id: شناسه آزمایش
    """
    try:
        # شروع آزمایش
        experiment_tracker.start_experiment(
            name=request.experiment_name,
            params={
                "model_type": request.model_type,
                "symbols": request.symbols,
                "target_column": request.target_column,
                "feature_columns": request.feature_columns,
                "hyperparameters": request.hyperparameters,
                "description": request.description
            }
        )
        
        # آماده‌سازی داده‌ها
        training_data = {}
        for symbol in request.symbols:
            try:
                data = data_manager.prepare_training_data(
                    symbol=symbol,
                    sequence_length=request.hyperparameters.get("sequence_length", 20),
                    target_column=request.target_column,
                    feature_columns=request.feature_columns,
                    train_split=request.hyperparameters.get("train_split", 0.7),
                    val_split=request.hyperparameters.get("val_split", 0.15)
                )
                training_data[symbol] = data
            except Exception as e:
                experiment_tracker.log_metric(f"error_{symbol}", 1.0)
                active_trainings[experiment_id]["status"] = f"خطا در آماده‌سازی داده {symbol}: {e}"
        
        # ایجاد و تنظیم مدل
        model_config = {
            "input_channels": 1 if request.hyperparameters.get("grayscale", True) else 3,
            "cnn_layers": request.hyperparameters.get("cnn_layers", [
                {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1}
            ]),
            "lstm_hidden_size": request.hyperparameters.get("lstm_hidden_size", 256),
            "lstm_num_layers": request.hyperparameters.get("lstm_num_layers", 2),
            "num_classes": request.hyperparameters.get("num_classes", 10),
            "dropout_rate": request.hyperparameters.get("dropout_rate", 0.5),
            "learning_rate": request.hyperparameters.get("learning_rate", 0.001)
        }
        
        model = CNNLSTMPatternModel(model_config)
        
        # تنظیم آموزش‌دهنده
        trainer_config = {
            "batch_size": request.hyperparameters.get("batch_size", 32),
            "num_epochs": request.hyperparameters.get("num_epochs", 100),
            "early_stopping_patience": request.hyperparameters.get("early_stopping_patience", 10),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "checkpoint_dir": config["experiment_tracking"]["checkpoints_dir"]
        }
        
        trainer = ModelTrainer(model, trainer_config)
        
        # آماده‌سازی داده‌ها برای آموزش
        # این بخش نیاز به پیاده‌سازی دارد تا داده‌ها را به DataLoader تبدیل کند
        # اینجا یک پیاده‌سازی ساده برای نمونه آورده شده است
        # در پروژه واقعی، این بخش باید با توجه به ساختار داده‌ها پیاده‌سازی شود
        
        # آموزش مدل
        active_trainings[experiment_id]["status"] = "در حال آموزش"
        history = trainer.train(train_loader=None, val_loader=None)  # باید DataLoader ایجاد شود
        
        # ذخیره مدل
        model_id = f"{request.model_name}_{experiment_id}"
        model_manager.save_model(model, request.model_name, {
            "experiment_id": experiment_id,
            "history": history,
            "config": model_config
        })
        
        # پایان آزمایش
        experiment_tracker.end_experiment()
        
        # به‌روزرسانی وضعیت
        active_trainings[experiment_id]["status"] = "تکمیل شده"
        active_trainings[experiment_id]["model_id"] = model_id
        active_trainings[experiment_id]["history"] = history
        
    except Exception as e:
        # ثبت خطا
        active_trainings[experiment_id]["status"] = f"خطا: {str(e)}"
        experiment_tracker.log_metric("error", 1.0)
        experiment_tracker.end_experiment()


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    API برای شروع آموزش یک مدل جدید.
    
    Args:
        request: پارامترهای آموزش
        background_tasks: وظایف پس‌زمینه
        
    Returns:
        وضعیت آموزش
    """
    try:
        # ایجاد شناسه آزمایش
        import time
        experiment_id = f"{request.model_name}_{int(time.time())}"
        
        # تنظیم آموزش در پس‌زمینه
        active_trainings[experiment_id] = {
            "model_name": request.model_name,
            "status": "در صف",
            "start_time": time.time(),
            "request": request.dict()
        }
        
        # شروع آموزش در پس‌زمینه
        background_tasks.add_task(train_model_task, request, experiment_id)
        
        return TrainingResponse(
            success=True,
            message="آموزش با موفقیت شروع شد",
            experiment_id=experiment_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در شروع آموزش: {str(e)}")


@router.get("/training/{experiment_id}/status")
async def get_training_status(experiment_id: str):
    """
    وضعیت آموزش را بررسی کنید.
    
    Args:
        experiment_id: شناسه آزمایش
        
    Returns:
        وضعیت آموزش
    """
    if experiment_id not in active_trainings:
        raise HTTPException(status_code=404, detail=f"آزمایش با شناسه {experiment_id} یافت نشد")
    
    return {
        "experiment_id": experiment_id,
        "status": active_trainings[experiment_id]["status"],
        "model_name": active_trainings[experiment_id]["model_name"],
        "elapsed_time": time.time() - active_trainings[experiment_id]["start_time"],
        "model_id": active_trainings[experiment_id].get("model_id")
    }


@router.get("/list")
async def list_models():
    """
    لیست مدل‌های موجود را بازگرداند.
    
    Returns:
        لیست مدل‌ها
    """
    try:
        models = model_manager.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در بازیابی لیست مدل‌ها: {str(e)}")


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """
    اطلاعات یک مدل خاص را بازگرداند.
    
    Args:
        model_id: شناسه مدل
        
    Returns:
        اطلاعات مدل
    """
    try:
        # یافتن مدل بر اساس شناسه
        models = model_manager.list_models()
        model_info = None
        
        for model in models:
            if model["name"] == model_id or f"{model['name']}_{model['timestamp']}" == model_id:
                model_info = model
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"مدل با شناسه {model_id} یافت نشد")
        
        return model_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در بازیابی اطلاعات مدل: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    با استفاده از یک مدل، پیش‌بینی انجام دهید.
    
    Args:
        request: پارامترهای پیش‌بینی
        
    Returns:
        نتایج پیش‌بینی
    """
    try:
        # یافتن مدل
        models = model_manager.list_models()
        model_info = None
        
        for model in models:
            if model["name"] == request.model_id or f"{model['name']}_{model['timestamp']}" == request.model_id:
                model_info = model
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"مدل با شناسه {request.model_id} یافت نشد")
        
        # بارگذاری مدل
        model_config = model_info["metadata"]["config"]
        model = CNNLSTMPatternModel(model_config)
        model_manager.load_model(model, model_info["name"], model_info["timestamp"])
        model.eval()
        
        # پیش‌بینی براساس نوع ورودی (تصویر یا داده)
        if request.image_url:
            # پیش‌بینی براساس تصویر
            import requests
            from PIL import Image
            from io import BytesIO
            import torchvision.transforms as transforms
            
            # دانلود تصویر
            response = requests.get(request.image_url)
            img = Image.open(BytesIO(response.content))
            
            # پیش‌پردازش تصویر
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale() if model_config["input_channels"] == 1 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            input_tensor = transform(img).unsqueeze(0)  # افزودن بعد batch
            
            # پیش‌بینی
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predictions = torch.max(probs, 1)
            
            # تبدیل به لیست نتایج
            results = [
                {
                    "class": int(predictions[0].item()),
                    "confidence": float(confidence[0].item())
                }
            ]
        
        elif request.data:
            # پیش‌بینی براساس داده‌های عددی
            # تبدیل داده‌ها به تنسور
            data_tensors = []
            for column, values in request.data.items():
                data_tensors.append(torch.tensor(values, dtype=torch.float32).view(-1, 1))
            
            input_tensor = torch.cat(data_tensors, dim=1).unsqueeze(0)  # افزودن بعد batch
            
            # پیش‌بینی
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predictions = torch.max(probs, 1)
            
            # تبدیل به لیست نتایج
            results = [
                {
                    "class": int(predictions[0].item()),
                    "confidence": float(confidence[0].item())
                }
            ]
        
        else:
            raise HTTPException(status_code=400, detail="باید یکی از پارامترهای image_url یا data را ارائه دهید")
        
        return PredictionResponse(
            success=True,
            predictions=results,
            confidence=float(confidence[0].item()),
            message="پیش‌بینی با موفقیت انجام شد"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در انجام پیش‌بینی: {str(e)}")


@router.post("/upload-image")
async def upload_image_for_prediction(
    file: UploadFile = File(...),
    model_id: str = None
):
    """
    آپلود تصویر برای پیش‌بینی.
    
    Args:
        file: فایل تصویر
        model_id: شناسه مدل (اختیاری)
        
    Returns:
        نتایج پیش‌بینی
    """
    try:
        # ذخیره تصویر
        import uuid
        import os
        
        # ایجاد دایرکتوری موقت اگر وجود ندارد
        os.makedirs("temp", exist_ok=True)
        
        # ذخیره فایل با یک نام تصادفی
        file_extension = os.path.splitext(file.filename)[1]
        temp_file = f"temp/{uuid.uuid4()}{file_extension}"
        
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        # اگر model_id مشخص نشده باشد، از آخرین مدل استفاده می‌کنیم
        if not model_id:
            models = model_manager.list_models()
            if not models:
                raise HTTPException(status_code=404, detail="هیچ مدلی یافت نشد")
            model_id = models[0]["name"]
        
        # ایجاد درخواست پیش‌بینی
        image_url = f"file://{os.path.abspath(temp_file)}"
        request = PredictionRequest(model_id=model_id, image_url=image_url)
        
        # انجام پیش‌بینی
        response = await predict(request)
        
        # پاکسازی فایل موقت
        os.remove(temp_file)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پیش‌بینی با تصویر: {str(e)}") 