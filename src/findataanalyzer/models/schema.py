"""Database schema definition using SQLAlchemy."""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User table."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(100))
    disabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    data_sources = relationship("DataSource", back_populates="owner")
    analyses = relationship("Analysis", back_populates="user")


class DataSource(Base):
    """Data source table."""
    
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    source_type = Column(String(20), nullable=False)  # file, url, api, etc.
    path = Column(String(500), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime)
    
    # Relationships
    owner = relationship("User", back_populates="data_sources")
    analyses = relationship("Analysis", back_populates="data_source")


class Analysis(Base):
    """Analysis results table."""
    
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    parameters = Column(JSON)  # Store analysis parameters
    results = Column(JSON)  # Store analysis results
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    data_source = relationship("DataSource", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
    predictions = relationship("Prediction", back_populates="analysis")


class Prediction(Base):
    """Prediction results table."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    target_column = Column(String(100), nullable=False)
    method = Column(String(50), nullable=False)  # linear, arima, etc.
    parameters = Column(JSON)  # Store prediction parameters
    results = Column(JSON)  # Store prediction results
    metrics = Column(JSON)  # Store evaluation metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="predictions")


class TimeSeriesData(Base):
    """Time series data table."""
    
    __tablename__ = "time_series_data"
    
    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"))
    timestamp = Column(DateTime, nullable=False)
    series_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    
    # Add index for efficient time series queries
    __table_args__ = ({"sqlite_autoincrement": True},) 