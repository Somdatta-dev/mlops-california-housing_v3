"""
FastAPI application configuration management.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import logging


class APIConfig(BaseSettings):
    """
    API configuration with environment variable support.
    """
    # API Settings
    app_name: str = Field(default="California Housing Price Prediction API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="MLOps API for predicting California housing prices using GPU-accelerated models",
        env="APP_DESCRIPTION"
    )
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    workers: int = Field(default=1, env="WORKERS")
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # Model Settings
    default_model_name: str = Field(default="linear_regression", env="DEFAULT_MODEL_NAME")
    model_cache_size: int = Field(default=3, env="MODEL_CACHE_SIZE")
    model_timeout: float = Field(default=30.0, env="MODEL_TIMEOUT")
    
    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="california_housing_prediction", 
        env="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_model_stage: str = Field(default="None", env="MLFLOW_MODEL_STAGE")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///./api_logs.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default="logs/api.log", env="LOG_FILE")
    log_rotation: str = Field(default="1 day", env="LOG_ROTATION")
    log_retention: str = Field(default="30 days", env="LOG_RETENTION")
    
    # Monitoring Settings
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH")
    
    # GPU Settings
    enable_gpu_monitoring: bool = Field(default=True, env="ENABLE_GPU_MONITORING")
    gpu_memory_limit: Optional[float] = Field(default=0.8, env="GPU_MEMORY_LIMIT")
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds
    
    # Batch Processing
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    batch_timeout: float = Field(default=60.0, env="BATCH_TIMEOUT")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: List[str] = Field(default=[], env="API_KEYS")
    
    # Health Check
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator('cors_methods', pre=True)
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator('cors_headers', pre=True)
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @validator('api_keys', pre=True)
    def parse_api_keys(cls, v):
        """Parse API keys from string or list."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()
    
    @validator('gpu_memory_limit')
    def validate_gpu_memory_limit(cls, v):
        """Validate GPU memory limit."""
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("GPU memory limit must be between 0.0 and 1.0")
        return v
    
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_credentials,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers,
        }
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
        }
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file": self.log_file,
            "rotation": self.log_rotation,
            "retention": self.log_retention,
        }


# Global configuration instance
config = APIConfig()


def get_config() -> APIConfig:
    """Get the global configuration instance."""
    return config


def setup_environment():
    """Setup environment variables and create necessary directories."""
    # Create logs directory if it doesn't exist
    if config.log_file:
        log_dir = os.path.dirname(config.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    return config


if __name__ == "__main__":
    # Test configuration loading
    config = setup_environment()
    print(f"API Configuration loaded:")
    print(f"  App Name: {config.app_name}")
    print(f"  Version: {config.app_version}")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Debug: {config.debug}")
    print(f"  Log Level: {config.log_level}")
    print(f"  MLflow URI: {config.mlflow_tracking_uri}")
    print(f"  Database: {config.database_url}") 