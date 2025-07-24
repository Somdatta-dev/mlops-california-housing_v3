"""
SQLAlchemy models for MLOps database.

This module defines database models for tracking predictions, performance metrics,
system health, and model versions in the MLOps pipeline.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    Index, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class ModelVersion(Base):
    """Track model versions and metadata."""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    mlflow_run_id = Column(String(100), nullable=True)
    
    # Model performance metrics
    training_r2 = Column(Float, nullable=True)
    training_rmse = Column(Float, nullable=True) 
    training_mae = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    test_rmse = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    
    # Model metadata
    feature_names = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    training_duration = Column(Float, nullable=True)  # seconds
    model_size_mb = Column(Float, nullable=True)
    gpu_accelerated = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Relationships
    predictions = relationship("PredictionLog", back_populates="model_version")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_version'),
        Index('idx_mv_model_name', 'model_name'),
        Index('idx_mv_created_at', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'version': self.version,
            'algorithm': self.algorithm,
            'mlflow_run_id': self.mlflow_run_id,
            'training_r2': self.training_r2,
            'training_rmse': self.training_rmse,
            'training_mae': self.training_mae,
            'test_r2': self.test_r2,
            'test_rmse': self.test_rmse,
            'test_mae': self.test_mae,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'training_duration': self.training_duration,
            'model_size_mb': self.model_size_mb,
            'gpu_accelerated': self.gpu_accelerated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class PredictionLog(Base):
    """Log all prediction requests and responses."""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(100), nullable=False, unique=True)
    
    # Model information
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Request details
    input_features = Column(JSON, nullable=False)
    prediction_value = Column(Float, nullable=True)
    prediction_confidence = Column(Float, nullable=True)
    
    # Request metadata
    request_method = Column(String(10), nullable=False)  # POST, GET
    endpoint = Column(String(200), nullable=False)
    user_agent = Column(String(500), nullable=True)
    client_ip = Column(String(50), nullable=True)
    
    # Performance metrics
    processing_time_ms = Column(Float, nullable=True)
    model_load_time_ms = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    
    # Response details
    status_code = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="predictions")
    
    # Constraints and Indexes
    __table_args__ = (
        Index('idx_pl_request_id', 'request_id'),
        Index('idx_pl_model_name', 'model_name'),
        Index('idx_pl_created_at', 'created_at'),
        Index('idx_pl_status_code', 'status_code'),
        Index('idx_pl_model_version_id', 'model_version_id'),
        CheckConstraint('status_code >= 100 AND status_code < 600', name='ck_valid_status_code'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction log to dictionary."""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'model_version_id': self.model_version_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'input_features': self.input_features,
            'prediction_value': self.prediction_value,
            'prediction_confidence': self.prediction_confidence,
            'request_method': self.request_method,
            'endpoint': self.endpoint,
            'user_agent': self.user_agent,
            'client_ip': self.client_ip,
            'processing_time_ms': self.processing_time_ms,
            'model_load_time_ms': self.model_load_time_ms,
            'inference_time_ms': self.inference_time_ms,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'response_size_bytes': self.response_size_bytes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class PerformanceMetrics(Base):
    """Track system and model performance metrics over time."""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    labels = Column(JSON, nullable=True)  # Additional labels as key-value pairs
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    
    # Context
    model_name = Column(String(100), nullable=True)
    endpoint = Column(String(200), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Constraints and Indexes
    __table_args__ = (
        Index('idx_pm_metric_name', 'metric_name'),
        Index('idx_pm_metric_type', 'metric_type'),
        Index('idx_pm_timestamp', 'timestamp'),
        Index('idx_pm_model_name', 'model_name'),
        Index('idx_pm_composite', 'metric_name', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metric to dictionary."""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'labels': self.labels,
            'value': self.value,
            'unit': self.unit,
            'model_name': self.model_name,
            'endpoint': self.endpoint,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class SystemHealth(Base):
    """Track system health and resource utilization."""
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # System metrics
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    memory_used_mb = Column(Float, nullable=True)
    memory_total_mb = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)
    
    # API metrics
    active_requests = Column(Integer, nullable=True)
    total_requests_count = Column(Integer, nullable=True)
    error_rate_percent = Column(Float, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)
    
    # GPU metrics (if available)
    gpu_count = Column(Integer, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)
    gpu_memory_used_mb = Column(Float, nullable=True)
    gpu_memory_total_mb = Column(Float, nullable=True)
    gpu_temperature_c = Column(Float, nullable=True)
    
    # Model metrics
    models_loaded_count = Column(Integer, nullable=True)
    model_cache_size_mb = Column(Float, nullable=True)
    model_cache_hit_rate = Column(Float, nullable=True)
    
    # Health status
    status = Column(String(20), nullable=False, default='healthy')  # healthy, degraded, unhealthy
    status_message = Column(Text, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Constraints and Indexes
    __table_args__ = (
        Index('idx_sh_timestamp', 'timestamp'),
        Index('idx_sh_status', 'status'),
        CheckConstraint("status IN ('healthy', 'degraded', 'unhealthy')", name='ck_valid_status'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system health to dictionary."""
        return {
            'id': self.id,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'active_requests': self.active_requests,
            'total_requests_count': self.total_requests_count,
            'error_rate_percent': self.error_rate_percent,
            'avg_response_time_ms': self.avg_response_time_ms,
            'gpu_count': self.gpu_count,
            'gpu_usage_percent': self.gpu_usage_percent,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_temperature_c': self.gpu_temperature_c,
            'models_loaded_count': self.models_loaded_count,
            'model_cache_size_mb': self.model_cache_size_mb,
            'model_cache_hit_rate': self.model_cache_hit_rate,
            'status': self.status,
            'status_message': self.status_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        } 