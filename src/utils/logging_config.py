"""
Logging configuration for the MLOps pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Custom log format string
        enable_console: Whether to enable console logging
    """
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Default log file
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"mlops_pipeline_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.handlers.clear()  # Clear any existing handlers
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use colored formatting for console if available
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                show_time=False
            )
            console_handler.setLevel(getattr(logging, log_level.upper()))
        except ImportError:
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))
        
        logger.addHandler(console_handler)
    
    # Configure specific loggers to avoid spam
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"{func_name} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"{func_name} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)


def setup_mlflow_logging():
    """Setup MLflow-specific logging configuration."""
    # Reduce MLflow logging verbosity
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)
    logging.getLogger("mlflow.store").setLevel(logging.WARNING)


def setup_gpu_logging():
    """Setup GPU-specific logging configuration."""
    # Configure CUDA/GPU related logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("cudf").setLevel(logging.WARNING)
    logging.getLogger("cuml").setLevel(logging.WARNING)
    
    # XGBoost GPU logging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # For performance


def setup_api_logging():
    """Setup API-specific logging configuration."""
    # Configure FastAPI and Uvicorn logging
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO) 