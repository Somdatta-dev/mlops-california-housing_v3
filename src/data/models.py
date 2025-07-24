"""
Pydantic models for California Housing data validation and serialization.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np


class CaliforniaHousingData(BaseModel):
    """
    Pydantic model for California Housing dataset with comprehensive validation.
    """
    med_inc: float = Field(
        ...,
        description="Median income in block group (in tens of thousands)",
        ge=0.0,
        le=15.0,
        alias="MedInc"
    )
    
    house_age: float = Field(
        ...,
        description="Median house age in block group",
        ge=1.0,
        le=52.0,
        alias="HouseAge"
    )
    
    ave_rooms: float = Field(
        ...,
        description="Average number of rooms per household",
        ge=0.8,
        le=141.0,
        alias="AveRooms"
    )
    
    ave_bedrms: float = Field(
        ...,
        description="Average number of bedrooms per household",
        ge=0.1,
        le=34.0,
        alias="AveBedrms"
    )
    
    population: float = Field(
        ...,
        description="Population of block group",
        ge=3.0,
        le=35682.0
    )
    
    ave_occup: float = Field(
        ...,
        description="Average number of household members",
        ge=0.69,
        le=1243.0,
        alias="AveOccup"
    )
    
    latitude: float = Field(
        ...,
        description="Latitude coordinate",
        ge=32.54,
        le=41.95
    )
    
    longitude: float = Field(
        ...,
        description="Longitude coordinate",
        ge=-124.35,
        le=-114.31
    )
    
    target: Optional[float] = Field(
        None,
        description="Median house value (target variable)",
        ge=0.14999,
        le=5.00001
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23,
                "target": 4.526
            }
        }
    }

    @validator('ave_bedrms')
    def validate_bedroom_ratio(cls, v, values):
        """Validate that average bedrooms doesn't exceed average rooms."""
        if 'ave_rooms' in values and v > values['ave_rooms']:
            raise ValueError(
                f"Average bedrooms ({v}) cannot exceed average rooms ({values['ave_rooms']})"
            )
        return v

    @validator('ave_occup')
    def validate_occupancy(cls, v, values):
        """Validate reasonable occupancy levels."""
        if v > 50:
            raise ValueError(f"Average occupancy ({v}) seems unreasonably high (>50)")
        return v

    @root_validator(skip_on_failure=True)
    def validate_california_coordinates(cls, values):
        """Validate that coordinates are within California boundaries."""
        lat, lon = values.get('latitude'), values.get('longitude')
        
        if lat and lon:
            # Rough California boundary check
            if not (32.5 <= lat <= 42.0 and -125.0 <= lon <= -114.0):
                raise ValueError(
                    f"Coordinates ({lat}, {lon}) are outside California boundaries"
                )
        
        return values


class HousingPredictionRequest(BaseModel):
    """
    Request model for housing price prediction API.
    """
    features: CaliforniaHousingData = Field(..., description="Housing features for prediction")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.024,
                    "Population": 322.0,
                    "AveOccup": 2.555,
                    "Latitude": 37.88,
                    "Longitude": -122.23
                }
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch housing price predictions.
    """
    features: List[CaliforniaHousingData] = Field(
        ...,
        description="List of housing features for batch prediction",
        min_items=1,
        max_items=1000
    )

    @validator('features')
    def validate_batch_size(cls, v):
        """Validate reasonable batch size."""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 records")
        return v


class PredictionResponse(BaseModel):
    """
    Response model for single prediction.
    """
    prediction: float = Field(..., description="Predicted house value")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    prediction_id: Optional[str] = Field(None, description="Unique prediction identifier")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="Prediction confidence interval"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 4.526,
                "model_name": "xgboost_gpu",
                "model_version": "1.0.0",
                "prediction_id": "pred_123456",
                "confidence_interval": {"lower": 4.2, "upper": 4.8}
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    batch_id: str = Field(..., description="Unique batch identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "prediction": 4.526,
                        "model_name": "xgboost_gpu",
                        "model_version": "1.0.0",
                        "prediction_id": "pred_123456"
                    }
                ],
                "batch_id": "batch_789",
                "processing_time": 0.045
            }
        }
    }


class ModelInfo(BaseModel):
    """
    Model information and metadata.
    """
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="ML algorithm used")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    training_date: str = Field(..., description="Model training date")
    gpu_accelerated: bool = Field(..., description="Whether model uses GPU acceleration")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "xgboost_gpu",
                "version": "1.0.0",
                "algorithm": "XGBoost",
                "metrics": {
                    "rmse": 0.52,
                    "mae": 0.38,
                    "r2": 0.83
                },
                "training_date": "2025-01-20T10:30:00Z",
                "gpu_accelerated": True
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Invalid input data",
                "error_code": "VALIDATION_ERROR",
                "details": {
                    "field": "MedInc",
                    "message": "Value must be between 0.0 and 15.0"
                }
            }
        }
    }


class DataQualityReport(BaseModel):
    """
    Data quality validation report.
    """
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    validation_errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of validation errors"
    )
    data_profile: Dict[str, Any] = Field(
        default_factory=dict, description="Basic data profiling statistics"
    )
    
    @property
    def quality_score(self) -> float:
        """Calculate data quality score as percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100.0
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_records": 1000,
                "valid_records": 985,
                "invalid_records": 15,
                "validation_errors": [
                    {"record_id": 123, "field": "MedInc", "error": "Value out of range"}
                ],
                "data_profile": {
                    "mean_med_inc": 3.87,
                    "std_med_inc": 1.90,
                    "missing_values": 0
                }
            }
        }
    } 