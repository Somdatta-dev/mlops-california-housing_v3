"""
FastAPI dependency injection for shared resources.

This module provides dependency injection functions for database sessions,
model loading, and other shared resources.
"""

import time
import uuid
from typing import Generator, Dict, Any, Optional
from fastapi import Request, Depends
from sqlalchemy.orm import Session

from ..database import get_db_session, prediction_logger, metrics_logger, health_logger
from .model_loader import get_loaded_model, LoadedModel
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    
    Yields:
        SQLAlchemy session
    """
    yield from get_db_session()


def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Extract request metadata for logging.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary with request metadata
    """
    return {
        "request_id": str(uuid.uuid4()),
        "method": request.method,
        "url": str(request.url),
        "endpoint": request.url.path,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": time.time()
    }


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        logger.debug(f"{self.operation_name} took {self.duration_ms:.2f}ms")


async def log_prediction_request(
    request_metadata: Dict[str, Any],
    model: LoadedModel,
    input_features: Dict[str, Any],
    prediction_value: Optional[float] = None,
    prediction_confidence: Optional[float] = None,
    processing_time_ms: Optional[float] = None,
    model_load_time_ms: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    status_code: int = 200,
    error_message: Optional[str] = None,
    response_size_bytes: Optional[int] = None
) -> bool:
    """
    Log prediction request to database.
    
    Args:
        request_metadata: Request metadata dictionary
        model: Loaded model information
        input_features: Input features for prediction
        prediction_value: Predicted value
        prediction_confidence: Confidence score
        processing_time_ms: Total processing time
        model_load_time_ms: Model loading time
        inference_time_ms: Inference time
        status_code: HTTP status code
        error_message: Error message if any
        response_size_bytes: Response size
        
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        return prediction_logger.log_prediction(
            request_id=request_metadata["request_id"],
            model_name=model.metadata.name,
            model_version=model.metadata.version,
            input_features=input_features,
            prediction_value=prediction_value,
            prediction_confidence=prediction_confidence,
            request_method=request_metadata["method"],
            endpoint=request_metadata["endpoint"],
            user_agent=request_metadata.get("user_agent"),
            client_ip=request_metadata.get("client_ip"),
            processing_time_ms=processing_time_ms,
            model_load_time_ms=model_load_time_ms,
            inference_time_ms=inference_time_ms,
            status_code=status_code,
            error_message=error_message,
            response_size_bytes=response_size_bytes
        )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        return False


async def log_metric(
    metric_name: str,
    value: float,
    metric_type: str = "gauge",
    labels: Optional[Dict[str, str]] = None,
    unit: Optional[str] = None,
    model_name: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Log performance metric to database.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        metric_type: Type of metric (counter, gauge, histogram)
        labels: Additional labels
        unit: Unit of measurement
        model_name: Associated model name
        endpoint: Associated endpoint
        
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        return metrics_logger.log_metric(
            metric_name=metric_name,
            value=value,
            metric_type=metric_type,
            labels=labels,
            unit=unit,
            model_name=model_name,
            endpoint=endpoint
        )
    except Exception as e:
        logger.error(f"Failed to log metric: {e}")
        return False


async def log_system_health(
    models_loaded_count: Optional[int] = None,
    model_cache_size_mb: Optional[float] = None,
    model_cache_hit_rate: Optional[float] = None,
    gpu_metrics: Optional[Dict[str, Any]] = None,
    api_metrics: Optional[Dict[str, Any]] = None,
    status: str = "healthy",
    status_message: Optional[str] = None
) -> bool:
    """
    Log system health to database.
    
    Args:
        models_loaded_count: Number of loaded models
        model_cache_size_mb: Model cache size in MB
        model_cache_hit_rate: Cache hit rate (0-1)
        gpu_metrics: GPU metrics dictionary
        api_metrics: API metrics dictionary
        status: Health status (healthy, degraded, unhealthy)
        status_message: Status message
        
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        return health_logger.log_system_health(
            models_loaded_count=models_loaded_count,
            model_cache_size_mb=model_cache_size_mb,
            model_cache_hit_rate=model_cache_hit_rate,
            gpu_metrics=gpu_metrics,
            api_metrics=api_metrics,
            status=status,
            status_message=status_message
        )
    except Exception as e:
        logger.error(f"Failed to log system health: {e}")
        return False 