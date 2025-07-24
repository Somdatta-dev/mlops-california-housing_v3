"""
Database logging utilities for MLOps pipeline.

This module provides utilities for logging predictions, performance metrics,
and system health to the database.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import psutil

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

try:
    from .models import PredictionLog, PerformanceMetrics, SystemHealth, ModelVersion
    from .connection import get_database_manager
    from ..utils.logging_config import get_logger
except ImportError:
    # For direct script execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from database.models import PredictionLog, PerformanceMetrics, SystemHealth, ModelVersion
    from database.connection import get_database_manager
    from utils.logging_config import get_logger

logger = get_logger(__name__)


class PredictionLogger:
    """Utility for logging prediction requests and responses."""
    
    def __init__(self):
        self.db_manager = None
    
    def _get_db_manager(self):
        """Lazy initialization of database manager."""
        if not self.db_manager:
            self.db_manager = get_database_manager()
        return self.db_manager
    
    def log_prediction(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction_value: Optional[float] = None,
        prediction_confidence: Optional[float] = None,
        request_method: str = "POST",
        endpoint: str = "/predict",
        user_agent: Optional[str] = None,
        client_ip: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        model_load_time_ms: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
        response_size_bytes: Optional[int] = None
    ) -> bool:
        """
        Log a prediction request to the database.
        
        Args:
            request_id: Unique request identifier
            model_name: Name of the model used
            model_version: Version of the model used
            input_features: Input features for prediction
            prediction_value: Predicted value
            prediction_confidence: Confidence score
            request_method: HTTP method
            endpoint: API endpoint
            user_agent: Client user agent
            client_ip: Client IP address
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
            with self._get_db_manager().get_session() as session:
                # Get model version ID if available
                model_version_id = None
                model_version_obj = session.query(ModelVersion).filter(
                    ModelVersion.model_name == model_name,
                    ModelVersion.version == model_version
                ).first()
                if model_version_obj:
                    model_version_id = model_version_obj.id
                
                # Create prediction log entry
                log_entry = PredictionLog(
                    request_id=request_id,
                    model_version_id=model_version_id,
                    model_name=model_name,
                    model_version=model_version,
                    input_features=input_features,
                    prediction_value=prediction_value,
                    prediction_confidence=prediction_confidence,
                    request_method=request_method,
                    endpoint=endpoint,
                    user_agent=user_agent,
                    client_ip=client_ip,
                    processing_time_ms=processing_time_ms,
                    model_load_time_ms=model_load_time_ms,
                    inference_time_ms=inference_time_ms,
                    status_code=status_code,
                    error_message=error_message,
                    response_size_bytes=response_size_bytes
                )
                
                session.add(log_entry)
                session.commit()
                
                logger.debug(f"Logged prediction: {request_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log prediction {request_id}: {e}")
            return False
    
    def get_recent_predictions(
        self, 
        limit: int = 100,
        model_name: Optional[str] = None,
        status_code: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent predictions from the database.
        
        Args:
            limit: Maximum number of records to return
            model_name: Filter by model name
            status_code: Filter by status code
            
        Returns:
            List of prediction records
        """
        try:
            with self._get_db_manager().get_session() as session:
                query = session.query(PredictionLog)
                
                if model_name:
                    query = query.filter(PredictionLog.model_name == model_name)
                if status_code:
                    query = query.filter(PredictionLog.status_code == status_code)
                
                predictions = query.order_by(
                    PredictionLog.created_at.desc()
                ).limit(limit).all()
                
                return [pred.to_dict() for pred in predictions]
                
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []


class MetricsLogger:
    """Utility for logging performance metrics."""
    
    def __init__(self):
        self.db_manager = None
    
    def _get_db_manager(self):
        """Lazy initialization of database manager."""
        if not self.db_manager:
            self.db_manager = get_database_manager()
        return self.db_manager
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> bool:
        """
        Log a performance metric.
        
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
            with self._get_db_manager().get_session() as session:
                metric = PerformanceMetrics(
                    metric_name=metric_name,
                    metric_type=metric_type,
                    labels=labels,
                    value=value,
                    unit=unit,
                    model_name=model_name,
                    endpoint=endpoint
                )
                
                session.add(metric)
                session.commit()
                
                logger.debug(f"Logged metric: {metric_name}={value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log metric {metric_name}: {e}")
            return False
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        model_name: Optional[str] = None,
        hours: int = 24,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get metrics from the database.
        
        Args:
            metric_name: Filter by metric name
            model_name: Filter by model name
            hours: Time window in hours
            limit: Maximum number of records
            
        Returns:
            List of metric records
        """
        try:
            with self._get_db_manager().get_session() as session:
                query = session.query(PerformanceMetrics)
                
                if metric_name:
                    query = query.filter(PerformanceMetrics.metric_name == metric_name)
                if model_name:
                    query = query.filter(PerformanceMetrics.model_name == model_name)
                
                # Time filter
                cutoff_time = datetime.now(timezone.utc) - timezone.timedelta(hours=hours)
                query = query.filter(PerformanceMetrics.timestamp >= cutoff_time)
                
                metrics = query.order_by(
                    PerformanceMetrics.timestamp.desc()
                ).limit(limit).all()
                
                return [metric.to_dict() for metric in metrics]
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []


class HealthLogger:
    """Utility for logging system health metrics."""
    
    def __init__(self):
        self.db_manager = None
    
    def _get_db_manager(self):
        """Lazy initialization of database manager."""
        if not self.db_manager:
            self.db_manager = get_database_manager()
        return self.db_manager
    
    def log_system_health(
        self,
        models_loaded_count: Optional[int] = None,
        model_cache_size_mb: Optional[float] = None,
        model_cache_hit_rate: Optional[float] = None,
        gpu_metrics: Optional[Dict[str, Any]] = None,
        api_metrics: Optional[Dict[str, Any]] = None,
        status: str = "healthy",
        status_message: Optional[str] = None
    ) -> bool:
        """
        Log system health metrics.
        
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
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Prepare GPU metrics
            gpu_count = None
            gpu_usage = None
            gpu_memory_used = None
            gpu_memory_total = None
            gpu_temperature = None
            
            if gpu_metrics:
                gpu_count = gpu_metrics.get('gpu_count')
                gpu_usage = gpu_metrics.get('utilization_percent')
                gpu_memory_used = gpu_metrics.get('memory_used_mb')
                gpu_memory_total = gpu_metrics.get('memory_total_mb')
                gpu_temperature = gpu_metrics.get('temperature_c')
            
            # Prepare API metrics
            active_requests = None
            total_requests = None
            error_rate = None
            avg_response_time = None
            
            if api_metrics:
                active_requests = api_metrics.get('active_requests')
                total_requests = api_metrics.get('total_requests')
                error_rate = api_metrics.get('error_rate_percent')
                avg_response_time = api_metrics.get('avg_response_time_ms')
            
            with self._get_db_manager().get_session() as session:
                health_entry = SystemHealth(
                    cpu_usage_percent=cpu_percent,
                    memory_usage_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_total_mb=memory.total / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    active_requests=active_requests,
                    total_requests_count=total_requests,
                    error_rate_percent=error_rate,
                    avg_response_time_ms=avg_response_time,
                    gpu_count=gpu_count,
                    gpu_usage_percent=gpu_usage,
                    gpu_memory_used_mb=gpu_memory_used,
                    gpu_memory_total_mb=gpu_memory_total,
                    gpu_temperature_c=gpu_temperature,
                    models_loaded_count=models_loaded_count,
                    model_cache_size_mb=model_cache_size_mb,
                    model_cache_hit_rate=model_cache_hit_rate,
                    status=status,
                    status_message=status_message
                )
                
                session.add(health_entry)
                session.commit()
                
                logger.debug(f"Logged system health: {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")
            return False
    
    def get_recent_health(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent system health records.
        
        Args:
            hours: Time window in hours
            limit: Maximum number of records
            
        Returns:
            List of health records
        """
        try:
            with self._get_db_manager().get_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timezone.timedelta(hours=hours)
                
                health_records = session.query(SystemHealth).filter(
                    SystemHealth.timestamp >= cutoff_time
                ).order_by(
                    SystemHealth.timestamp.desc()
                ).limit(limit).all()
                
                return [record.to_dict() for record in health_records]
                
        except Exception as e:
            logger.error(f"Failed to get health records: {e}")
            return []


# Global logger instances
prediction_logger = PredictionLogger()
metrics_logger = MetricsLogger()
health_logger = HealthLogger() 