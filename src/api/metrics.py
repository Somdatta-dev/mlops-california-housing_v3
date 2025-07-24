"""
Prometheus metrics for API monitoring.
"""

import time
import asyncio
from typing import Dict, Optional, Any, List
from functools import wraps
import psutil
import logging
from datetime import datetime

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
)

from ..utils.logging_config import get_logger
from .config import get_config

# Try to import GPU monitoring
try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    pynvml = None

logger = get_logger(__name__)
config = get_config()


class PrometheusMetrics:
    """Prometheus metrics collector for the API."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.gpu_available = GPU_AVAILABLE and config.enable_gpu_monitoring
        
        # Initialize GPU monitoring if available
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring enabled for {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_available = False
        
        self._setup_metrics()
        logger.info("Prometheus metrics initialized")
    
    def _setup_metrics(self):
        """Setup all Prometheus metrics."""
        
        # API Request Metrics
        self.request_count = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Prediction Metrics
        self.predictions_count = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['model_name', 'model_version', 'prediction_type'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'prediction_duration_seconds',
            'Prediction duration in seconds',
            ['model_name', 'model_version'],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
        )
        
        # Model Metrics
        self.models_loaded = Gauge(
            'models_loaded_count',
            'Number of models currently loaded',
            registry=self.registry
        )
        
        # System Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        # GPU Metrics (if available)
        if self.gpu_available:
            self.gpu_utilization = Gauge(
                'gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id', 'gpu_name'],
                registry=self.registry
            )
            
            self.gpu_memory_used = Gauge(
                'gpu_memory_used_bytes',
                'GPU memory used in bytes',
                ['gpu_id', 'gpu_name'],
                registry=self.registry
            )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_prediction(
        self, 
        model_name: str, 
        model_version: str, 
        duration: float,
        prediction_type: str = "single"
    ):
        """Record prediction metrics."""
        self.predictions_count.labels(
            model_name=model_name,
            model_version=model_version,
            prediction_type=prediction_type
        ).inc()
        
        self.prediction_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
    
    def update_system_metrics(self):
        """Update system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def update_gpu_metrics(self):
        """Update GPU metrics."""
        if not self.gpu_available:
            return
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU name
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.labels(
                    gpu_id=str(i),
                    gpu_name=name
                ).set(utilization.gpu)
                
                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_used.labels(
                    gpu_id=str(i),
                    gpu_name=name
                ).set(memory_info.used)
                
        except Exception as e:
            logger.warning(f"Failed to update GPU metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get formatted metrics for Prometheus."""
        return generate_latest(self.registry)


# Global metrics instance
_metrics: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
    return _metrics


async def start_metrics_collection():
    """Start background metrics collection."""
    metrics = get_metrics()
    
    async def collect_metrics():
        """Background task to collect system and GPU metrics."""
        while True:
            try:
                metrics.update_system_metrics()
                metrics.update_gpu_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    # Start the background task
    asyncio.create_task(collect_metrics())
    logger.info("Started metrics collection background task") 