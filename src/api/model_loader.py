"""
Model loading utilities with MLflow integration and caching.
"""

import os
import json
import pickle
import joblib
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from threading import Lock
import hashlib

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
import pandas as pd
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.mlflow_config import MLflowConfig
from .config import get_config
from .exceptions import ModelNotFoundError, ModelLoadError, ConfigurationError


logger = get_logger(__name__)
config = get_config()


@dataclass
class ModelMetadata:
    """Model metadata container."""
    name: str
    version: str
    algorithm: str
    mlflow_run_id: str
    artifact_path: str
    model_path: Path
    scaler_path: Optional[Path] = None
    feature_names: Optional[List[str]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    training_date: Optional[datetime] = None
    gpu_accelerated: bool = False
    model_size: Optional[int] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "mlflow_run_id": self.mlflow_run_id,
            "metrics": self.metrics,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "gpu_accelerated": self.gpu_accelerated,
            "model_size": self.model_size,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }


@dataclass
class LoadedModel:
    """Container for loaded model and associated components."""
    model: Any
    scaler: Optional[Any] = None
    feature_names: Optional[List[str]] = None
    metadata: Optional[ModelMetadata] = None
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions with the loaded model."""
        # Ensure input is properly formatted
        if isinstance(X, pd.DataFrame):
            # Reorder columns if feature names are available
            if self.feature_names:
                X = X[self.feature_names]
            X = X.values
        
        # Apply scaling if available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Make prediction
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        if hasattr(self.model, 'predict_proba'):
            if isinstance(X, pd.DataFrame):
                if self.feature_names:
                    X = X[self.feature_names]
                X = X.values
            
            if self.scaler:
                X = self.scaler.transform(X)
            
            return self.model.predict_proba(X)
        return None


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache: Dict[str, LoadedModel] = {}
        self.access_order: List[str] = []
        self.lock = Lock()
        logger.info(f"Initialized model cache with max size: {max_size}")
    
    def get(self, model_key: str) -> Optional[LoadedModel]:
        """Get model from cache."""
        with self.lock:
            if model_key in self.cache:
                # Update access order
                self.access_order.remove(model_key)
                self.access_order.append(model_key)
                
                # Update metadata
                model = self.cache[model_key]
                if model.metadata:
                    model.metadata.last_accessed = datetime.now()
                    model.metadata.access_count += 1
                
                logger.debug(f"Cache hit for model: {model_key}")
                return model
        
        logger.debug(f"Cache miss for model: {model_key}")
        return None
    
    def put(self, model_key: str, model: LoadedModel) -> None:
        """Put model in cache with LRU eviction."""
        with self.lock:
            # Remove if already exists
            if model_key in self.cache:
                self.access_order.remove(model_key)
            
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                logger.debug(f"Evicted model from cache: {oldest_key}")
            
            # Add new model
            self.cache[model_key] = model
            self.access_order.append(model_key)
            logger.info(f"Cached model: {model_key}")
    
    def remove(self, model_key: str) -> bool:
        """Remove model from cache."""
        with self.lock:
            if model_key in self.cache:
                del self.cache[model_key]
                self.access_order.remove(model_key)
                logger.info(f"Removed model from cache: {model_key}")
                return True
        return False
    
    def clear(self) -> None:
        """Clear all models from cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Cleared model cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(
                model.metadata.model_size or 0 
                for model in self.cache.values() 
                if model.metadata
            )
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_memory_mb": total_size / (1024 * 1024) if total_size else 0,
                "models": [
                    {
                        "key": key,
                        "name": model.metadata.name if model.metadata else "unknown",
                        "last_accessed": model.metadata.last_accessed.isoformat() if model.metadata else None,
                        "access_count": model.metadata.access_count if model.metadata else 0
                    }
                    for key, model in self.cache.items()
                ]
            }


class ModelLoader:
    """Model loader with MLflow integration."""
    
    def __init__(self, mlflow_config: Optional[MLflowConfig] = None):
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.cache = ModelCache(max_size=config.model_cache_size)
        self._available_models: Dict[str, ModelMetadata] = {}
        self._last_refresh = None
        logger.info("Initialized ModelLoader")
    
    def _generate_model_key(self, model_name: str, version: Optional[str] = None) -> str:
        """Generate cache key for model."""
        if version:
            return f"{model_name}:{version}"
        return model_name
    
    def _load_model_artifacts(self, model_name: str, version: str) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
        """Load model and associated artifacts from MLflow."""
        try:
            # Load the model using registered model URI
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
            
            # For our simple models, we don't have scalers
            scaler = None
            
            # Use standard feature names for California housing
            feature_names = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
            logger.info(f"Using standard feature names: {len(feature_names)} features")
            
            return model, scaler, feature_names
            
        except Exception as e:
            raise ModelLoadError(model_name, f"Failed to load model from registry: {e}")
    
    def _get_model_metadata(self, run_id: str) -> ModelMetadata:
        """Get model metadata from MLflow run."""
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            # Extract metadata
            name = run.data.tags.get("model_name", "unknown")
            version = run.data.tags.get("model_version", "1.0.0")
            algorithm = run.data.tags.get("algorithm", "unknown")
            artifact_path = run.data.tags.get("artifact_path", "model")
            
            # Extract metrics
            metrics = dict(run.data.metrics)
            
            # Check for GPU acceleration
            gpu_accelerated = run.data.tags.get("gpu_accelerated", "false").lower() == "true"
            
            # Parse training date
            training_date = None
            if run.info.start_time:
                training_date = datetime.fromtimestamp(run.info.start_time / 1000)
            
            # Get model size
            model_size = None
            try:
                artifacts = client.list_artifacts(run_id, artifact_path)
                if artifacts:
                    model_size = artifacts[0].file_size
            except Exception:
                pass
            
            return ModelMetadata(
                name=name,
                version=version,
                algorithm=algorithm,
                mlflow_run_id=run_id,
                artifact_path=artifact_path,
                model_path=Path(f"runs:/{run_id}/{artifact_path}"),
                metrics=metrics,
                training_date=training_date,
                gpu_accelerated=gpu_accelerated,
                model_size=model_size
            )
            
        except Exception as e:
            raise ModelLoadError("metadata", f"Failed to get metadata: {e}")
    
    async def refresh_available_models(self) -> None:
        """Refresh list of available models from MLflow."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get registered models instead of searching runs
            registered_models = client.search_registered_models()
            
            self._available_models.clear()
            
            for model in registered_models:
                try:
                    # Get the latest version
                    latest_versions = client.get_latest_versions(model.name, stages=["None"])
                    if latest_versions:
                        latest_version = latest_versions[0]
                        
                        # Create metadata from registered model
                        # Handle timestamp conversion
                        try:
                            if isinstance(latest_version.creation_timestamp, str):
                                training_date = datetime.fromisoformat(latest_version.creation_timestamp)
                            else:
                                training_date = datetime.fromtimestamp(latest_version.creation_timestamp / 1000)
                        except:
                            training_date = datetime.now()
                        
                        metadata = ModelMetadata(
                            name=model.name,
                            version=str(latest_version.version),  # Convert to string
                            algorithm=model.name.replace('_', ' ').title(),
                            mlflow_run_id=latest_version.run_id,
                            artifact_path=model.name,  # Use model name as artifact path
                            model_path=Path(f"models:/{model.name}/{latest_version.version}"),
                            training_date=training_date,  # Add this line that was missing
                            feature_names=[
                                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                'Population', 'AveOccup', 'Latitude', 'Longitude'
                            ]
                        )
                        
                        model_key = self._generate_model_key(metadata.name, metadata.version)
                        self._available_models[model_key] = metadata
                        logger.info(f"Found registered model: {model_key}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process registered model {model.name}: {e}")
            
            self._last_refresh = datetime.now()
            logger.info(f"Refreshed {len(self._available_models)} available models")
            
        except Exception as e:
            logger.error(f"Failed to refresh available models: {e}")
            # Don't raise exception, just log warning to allow graceful degradation
            logger.warning("Continuing with empty model list")
    
    async def load_model(
        self, 
        model_name: str, 
        version: Optional[str] = None, 
        force_reload: bool = False
    ) -> LoadedModel:
        """Load model from MLflow or cache."""
        model_key = self._generate_model_key(model_name, version)
        
        # Check cache first
        if not force_reload:
            cached_model = self.cache.get(model_key)
            if cached_model:
                logger.info(f"Loaded model from cache: {model_key}")
                return cached_model
        
        # Refresh available models if needed
        if not self._available_models or not self._last_refresh:
            await self.refresh_available_models()
        
        # Find model metadata
        if model_key not in self._available_models:
            # Try to find model with any version
            matching_keys = [k for k in self._available_models.keys() if k.startswith(f"{model_name}:")]
            if matching_keys:
                model_key = matching_keys[0]  # Use first matching version
                logger.info(f"Found model with key: {model_key}")
            else:
                available_keys = list(self._available_models.keys())
                logger.error(f"Model '{model_name}' not found. Available: {available_keys}")
                raise ModelNotFoundError(model_name, f"Model '{model_name}' not found")
        
        metadata = self._available_models[model_key]
        
        try:
            # Load model artifacts
            model, scaler, feature_names = self._load_model_artifacts(
                metadata.name, 
                metadata.version
            )
            
            # Create loaded model
            loaded_model = LoadedModel(
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                metadata=metadata
            )
            
            # Cache the model
            self.cache.put(model_key, loaded_model)
            
            logger.info(f"Successfully loaded model: {model_key}")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise ModelLoadError(model_name, str(e))
    
    async def get_default_model(self) -> LoadedModel:
        """Get the default model."""
        return await self.load_model(config.default_model_name)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        return [metadata.to_dict() for metadata in self._available_models.values()]
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return self.cache.get_cache_info()
    
    async def preload_models(self, model_names: List[str]) -> None:
        """Preload multiple models for faster access."""
        logger.info(f"Preloading {len(model_names)} models...")
        
        tasks = []
        for model_name in model_names:
            task = asyncio.create_task(self.load_model(model_name))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Successfully preloaded {successful}/{len(model_names)} models")


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


async def initialize_model_loader() -> ModelLoader:
    """Initialize and return the model loader."""
    loader = get_model_loader()
    await loader.refresh_available_models()
    return loader 