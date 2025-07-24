"""
Custom exceptions for the FastAPI application.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class APIException(HTTPException):
    """Base API exception class."""
    
    def __init__(
        self, 
        status_code: int, 
        detail: str, 
        error_code: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class ValidationError(APIException):
    """Validation error exception."""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )
        self.field = field


class ModelNotFoundError(APIException):
    """Model not found exception."""
    
    def __init__(self, model_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
            error_code="MODEL_NOT_FOUND"
        )
        self.model_name = model_name


class ModelLoadError(APIException):
    """Model loading error exception."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR"
        )
        self.model_name = model_name
        self.reason = reason


class PredictionError(APIException):
    """Prediction error exception."""
    
    def __init__(self, detail: str, model_name: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="PREDICTION_ERROR"
        )
        self.model_name = model_name


class GPUError(APIException):
    """GPU-related error exception."""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"GPU error: {detail}",
            error_code="GPU_ERROR"
        )


class DatabaseError(APIException):
    """Database error exception."""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {detail}",
            error_code="DATABASE_ERROR"
        )


class RateLimitError(APIException):
    """Rate limit exceeded exception."""
    
    def __init__(self, retry_after: Optional[int] = None):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            headers=headers
        )


class AuthenticationError(APIException):
    """Authentication error exception."""
    
    def __init__(self, detail: str = "Invalid API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            headers={"WWW-Authenticate": "Bearer"}
        )


class BatchSizeError(APIException):
    """Batch size error exception."""
    
    def __init__(self, current_size: int, max_size: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {current_size} exceeds maximum of {max_size}",
            error_code="BATCH_SIZE_ERROR"
        )
        self.current_size = current_size
        self.max_size = max_size


class TimeoutError(APIException):
    """Request timeout error exception."""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"{operation} timed out after {timeout} seconds",
            error_code="TIMEOUT_ERROR"
        )
        self.operation = operation
        self.timeout = timeout


class ServiceUnavailableError(APIException):
    """Service unavailable exception."""
    
    def __init__(self, detail: str, retry_after: Optional[int] = None):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
            headers=headers
        )


class ConfigurationError(APIException):
    """Configuration error exception."""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {detail}",
            error_code="CONFIGURATION_ERROR"
        )


# Exception mapping for better error handling
EXCEPTION_MAPPING = {
    ValueError: ValidationError,
    FileNotFoundError: ModelNotFoundError,
    RuntimeError: PredictionError,
    ConnectionError: DatabaseError,
    TimeoutError: TimeoutError,
}


def map_exception(exc: Exception) -> APIException:
    """Map standard exceptions to API exceptions."""
    exc_type = type(exc)
    
    if exc_type in EXCEPTION_MAPPING:
        api_exc_class = EXCEPTION_MAPPING[exc_type]
        return api_exc_class(str(exc))
    
    # Default to internal server error
    return APIException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
        error_code="INTERNAL_ERROR"
    ) 