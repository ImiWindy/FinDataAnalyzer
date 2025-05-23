"""Pydantic data models for the application."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime


class AnalysisRequest(BaseModel):
    """Request model for data analysis."""
    
    data_source: str = Field(..., description="Path or URL to the data source")
    parameters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Optional parameters for analysis"
    )
    
    @validator('data_source')
    def validate_data_source(cls, v):
        """Validate the data source."""
        if not v:
            raise ValueError("Data source cannot be empty")
        return v


class AnalysisResponse(BaseModel):
    """Response model for data analysis results."""
    
    success: bool = Field(..., description="Whether the analysis was successful")
    results: Dict[str, Any] = Field(
        default={},
        description="Analysis results"
    )
    message: str = Field(default="", description="Status message")


class TimeSeriesData(BaseModel):
    """Model for time series data."""
    
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    value: float = Field(..., description="Value at the timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    
    data_source: str = Field(..., description="Path or URL to the data source")
    target_column: str = Field(..., description="Column to predict")
    features: Optional[List[str]] = Field(
        default=None,
        description="Features to use for prediction"
    )
    method: str = Field(
        default="linear",
        description="Prediction method (linear, arima, etc.)"
    )
    horizon: int = Field(
        default=1,
        description="Number of steps to predict into the future"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    
    success: bool = Field(..., description="Whether the prediction was successful")
    method: str = Field(..., description="Prediction method used")
    target: str = Field(..., description="Target column predicted")
    predictions: Dict[str, Any] = Field(
        default={},
        description="Prediction results"
    )
    message: str = Field(default="", description="Status message")


class User(BaseModel):
    """User model."""
    
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    disabled: bool = Field(default=False, description="Whether the user is disabled")


class DataSource(BaseModel):
    """Data source model."""
    
    name: str = Field(..., description="Name of the data source")
    source_type: str = Field(..., description="Type of data source (file, url, api, etc.)")
    path: str = Field(..., description="Path or URL to the data source")
    description: Optional[str] = Field(default=None, description="Description of the data source")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 