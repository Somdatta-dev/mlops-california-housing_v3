"""
API endpoints for the FastAPI application.
"""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import Response
import pandas as pd
import numpy as np

from ..data.models import (
    HousingPredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    ErrorResponse
)
from ..utils.logging_config import get_logger
from .config import get_config
from .model_loader import get_model_loader, LoadedModel
from .exceptions import (
    ValidationError, ModelNotFoundError, PredictionError, 
    BatchSizeError, TimeoutError
)
from .metrics import get_metrics

logger = get_logger(__name__)
config = get_config()


# Create router
router = APIRouter()


async def get_loaded_model(model_name: Optional[str] = None) -> LoadedModel:
    """Dependency to get a loaded model."""
    model_loader = get_model_loader()
    
    if model_name:
        return await model_loader.load_model(model_name)
    else:
        return await model_loader.get_default_model()


def create_prediction_response(
    prediction: float,
    model: LoadedModel,
    prediction_id: Optional[str] = None
) -> PredictionResponse:
    """Create a prediction response."""
    return PredictionResponse(
        prediction=float(prediction),
        model_name=model.metadata.name if model.metadata else "unknown",
        model_version=model.metadata.version if model.metadata else "1.0.0",
        prediction_id=prediction_id or str(uuid.uuid4())
    )


@router.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    start_time = time.time()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.app_version,
        "uptime_check": True
    }
    
    try:
        # Check model loader
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        cache_info = model_loader.get_cache_info()
        
        health_data.update({
            "models": {
                "available_count": len(available_models),
                "cached_count": cache_info["size"],
                "default_model": config.default_model_name
            }
        })
        
        # Try to load default model (quick check)
        try:
            default_model = await asyncio.wait_for(
                model_loader.get_default_model(),
                timeout=5.0
            )
            health_data["models"]["default_model_loaded"] = True
        except asyncio.TimeoutError:
            health_data["models"]["default_model_loaded"] = False
            health_data["status"] = "degraded"
        except Exception as e:
            health_data["models"]["default_model_loaded"] = False
            health_data["models"]["default_model_error"] = str(e)
            health_data["status"] = "degraded"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_data.update({
            "status": "unhealthy",
            "error": str(e)
        })
    
    # Add response time
    health_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return health_data


@router.post(
    "/predict", 
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        500: {"model": ErrorResponse, "description": "Prediction Error"}
    },
    tags=["Predictions"]
)
async def predict_single(
    request: HousingPredictionRequest,
    model_name: Optional[str] = None,
    model: LoadedModel = Depends(get_loaded_model)
) -> PredictionResponse:
    """
    Make a single housing price prediction.
    
    This endpoint accepts housing features and returns a predicted house value
    using the specified model (or default model if not specified).
    """
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting prediction {prediction_id} with model {model.metadata.name if model.metadata else 'unknown'}")
        
        # Convert request to DataFrame
        features_dict = request.features.dict(by_alias=True)
        # Remove target if present
        features_dict.pop('target', None)
        
        df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Create response
        response = create_prediction_response(
            prediction=prediction[0],
            model=model,
            prediction_id=prediction_id
        )
        
        # Record metrics
        duration = time.time() - start_time
        metrics = get_metrics()
        metrics.record_prediction(
            model_name=response.model_name,
            model_version=response.model_version,
            duration=duration,
            prediction_type="single"
        )
        
        logger.info(f"Prediction {prediction_id} completed in {duration:.3f}s: {prediction[0]:.3f}")
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Prediction {prediction_id} failed after {duration:.3f}s: {e}")
        
        # Record error metrics
        metrics = get_metrics()
        if model.metadata:
            metrics.record_prediction_error(model.metadata.name, type(e).__name__)
        
        if isinstance(e, (ValidationError, ModelNotFoundError)):
            raise e
        else:
            raise PredictionError(f"Prediction failed: {str(e)}")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Batch Too Large"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        500: {"model": ErrorResponse, "description": "Prediction Error"}
    },
    tags=["Predictions"]
)
async def predict_batch(
    request: BatchPredictionRequest,
    model_name: Optional[str] = None,
    model: LoadedModel = Depends(get_loaded_model)
) -> BatchPredictionResponse:
    """
    Make batch housing price predictions.
    
    This endpoint accepts a list of housing features and returns predicted house values
    for each input using the specified model (or default model if not specified).
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    batch_size = len(request.features)
    
    # Check batch size
    if batch_size > config.max_batch_size:
        raise BatchSizeError(batch_size, config.max_batch_size)
    
    try:
        logger.info(f"Starting batch prediction {batch_id} with {batch_size} samples using model {model.metadata.name if model.metadata else 'unknown'}")
        
        # Convert requests to DataFrame
        features_list = []
        for features in request.features:
            features_dict = features.dict(by_alias=True)
            # Remove target if present
            features_dict.pop('target', None)
            features_list.append(features_dict)
        
        df = pd.DataFrame(features_list)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Create individual prediction responses
        prediction_responses = []
        for i, prediction in enumerate(predictions):
            pred_response = create_prediction_response(
                prediction=prediction,
                model=model,
                prediction_id=f"{batch_id}_{i}"
            )
            prediction_responses.append(pred_response)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create batch response
        response = BatchPredictionResponse(
            predictions=prediction_responses,
            batch_id=batch_id,
            processing_time=processing_time
        )
        
        # Record metrics
        metrics = get_metrics()
        metrics.record_prediction(
            model_name=response.predictions[0].model_name,
            model_version=response.predictions[0].model_version,
            duration=processing_time,
            prediction_type="batch"
        )
        
        logger.info(f"Batch prediction {batch_id} completed in {processing_time:.3f}s for {batch_size} samples")
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Batch prediction {batch_id} failed after {duration:.3f}s: {e}")
        
        # Record error metrics
        metrics = get_metrics()
        if model.metadata:
            metrics.record_prediction_error(model.metadata.name, type(e).__name__)
        
        if isinstance(e, (ValidationError, ModelNotFoundError, BatchSizeError)):
            raise e
        else:
            raise PredictionError(f"Batch prediction failed: {str(e)}")


@router.get(
    "/models",
    response_model=List[ModelInfo],
    tags=["Models"]
)
async def list_models() -> List[ModelInfo]:
    """
    List all available models.
    
    Returns metadata for all models available in the MLflow registry.
    """
    try:
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        
        model_infos = []
        for model_data in available_models:
            model_info = ModelInfo(
                name=model_data["name"],
                version=model_data["version"],
                algorithm=model_data["algorithm"],
                metrics=model_data["metrics"],
                training_date=model_data["training_date"] or datetime.utcnow().isoformat(),
                gpu_accelerated=model_data["gpu_accelerated"]
            )
            model_infos.append(model_info)
        
        return model_infos
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get(
    "/models/{model_name}",
    response_model=ModelInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Model Not Found"}
    },
    tags=["Models"]
)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get information about a specific model.
    
    Returns detailed metadata for the specified model.
    """
    try:
        model_loader = get_model_loader()
        model = await model_loader.load_model(model_name)
        
        if not model.metadata:
            raise ModelNotFoundError(model_name)
        
        return ModelInfo(
            name=model.metadata.name,
            version=model.metadata.version,
            algorithm=model.metadata.algorithm,
            metrics=model.metadata.metrics,
            training_date=model.metadata.training_date.isoformat() if model.metadata.training_date else datetime.utcnow().isoformat(),
            gpu_accelerated=model.metadata.gpu_accelerated
        )
        
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")


@router.get(
    "/models/cache/info",
    tags=["Models", "System"]
)
async def get_cache_info() -> Dict[str, Any]:
    """
    Get model cache information.
    
    Returns statistics about the model cache including loaded models and memory usage.
    """
    try:
        model_loader = get_model_loader()
        cache_info = model_loader.get_cache_info()
        return cache_info
        
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cache info: {str(e)}")


@router.post(
    "/models/cache/clear",
    tags=["Models", "System"]
)
async def clear_cache() -> Dict[str, str]:
    """
    Clear the model cache.
    
    Removes all loaded models from memory cache. Models will be reloaded on next request.
    """
    try:
        model_loader = get_model_loader()
        model_loader.cache.clear()
        
        logger.info("Model cache cleared")
        return {"message": "Model cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get(
    "/system/status",
    tags=["System"]
)
async def system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.
    
    Returns detailed information about the system, models, and performance metrics.
    """
    try:
        model_loader = get_model_loader()
        metrics = get_metrics()
        
        # Get basic system info
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": config.app_version,
            "environment": {
                "debug": config.debug,
                "gpu_monitoring": config.enable_gpu_monitoring,
                "prometheus": config.enable_prometheus,
                "rate_limiting": config.enable_rate_limiting
            }
        }
        
        # Add model information
        available_models = model_loader.get_available_models()
        cache_info = model_loader.get_cache_info()
        
        status["models"] = {
            "available": len(available_models),
            "cached": cache_info["size"],
            "cache_memory_mb": cache_info["total_memory_mb"],
            "default_model": config.default_model_name
        }
        
        # Add configuration info
        status["configuration"] = {
            "max_batch_size": config.max_batch_size,
            "model_cache_size": config.model_cache_size,
            "rate_limit_requests": config.rate_limit_requests if config.enable_rate_limiting else None,
            "rate_limit_period": config.rate_limit_period if config.enable_rate_limiting else None
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {str(e)}")


# Metrics endpoint for Prometheus
@router.get(
    "/metrics",
    tags=["System"],
    include_in_schema=False  # Don't include in OpenAPI schema
)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not config.enable_prometheus:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
    
    metrics = get_metrics()
    metrics_data = metrics.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    ) 