"""
Middleware for the FastAPI application.
"""

import time
import json
import uuid
import asyncio
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from ..utils.logging_config import get_logger
from .config import get_config
from .exceptions import APIException, RateLimitError, map_exception
from .metrics import get_metrics

logger = get_logger(__name__)
config = get_config()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "user_agent": request.headers.get("user-agent"),
            "remote_addr": request.client.host if request.client else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log request body for POST/PUT requests (with size limit)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) < 10000:  # Only log if body is less than 10KB
                    request_data["body_size"] = len(body)
                    if body:
                        try:
                            request_data["body"] = json.loads(body.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_data["body"] = "<binary or invalid json>"
                else:
                    request_data["body_size"] = len(body)
                    request_data["body"] = "<body too large to log>"
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        logger.info(f"Request started", extra={"request_data": request_data})
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": len(response.body) if hasattr(response, 'body') else None
            }
            
            if response.status_code >= 400:
                logger.warning(f"Request completed with error", extra={"response_data": response_data})
            else:
                logger.info(f"Request completed", extra={"response_data": response_data})
            
            # Record metrics
            metrics = get_metrics()
            metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            error_data = {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": round(duration * 1000, 2)
            }
            
            logger.error(f"Request failed", extra={"error_data": error_data})
            
            # Record error metrics
            metrics = get_metrics()
            status_code = getattr(e, 'status_code', 500)
            metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=duration
            )
            
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except APIException as e:
            # API exceptions are already properly formatted
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "error_code": e.error_code,
                    "request_id": getattr(request.state, 'request_id', None)
                },
                headers=e.headers
            )
        except HTTPException as e:
            # FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "error_code": "HTTP_ERROR",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
        except Exception as e:
            # Map other exceptions to API exceptions
            api_exception = map_exception(e)
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=api_exception.status_code,
                content={
                    "error": api_exception.detail,
                    "error_code": api_exception.error_code,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.requests: Dict[str, list] = {}
        self.enabled = config.enable_rate_limiting
        self.max_requests = config.rate_limit_requests
        self.window_seconds = config.rate_limit_period
        
        if self.enabled:
            logger.info(f"Rate limiting enabled: {self.max_requests} requests per {self.window_seconds} seconds")
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if available, otherwise IP address
        api_key = request.headers.get(config.api_key_header)
        if api_key:
            return f"api_key:{api_key}"
        
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _cleanup_old_requests(self, client_id: str) -> None:
        """Remove old requests outside the time window."""
        if client_id not in self.requests:
            return
        
        cutoff_time = time.time() - self.window_seconds
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if timestamp > cutoff_time
        ]
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        self._cleanup_old_requests(client_id)
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        return len(self.requests[client_id]) >= self.max_requests
    
    def _record_request(self, client_id: str) -> None:
        """Record a new request for the client."""
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id].append(time.time())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        if self._is_rate_limited(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise RateLimitError(retry_after=self.window_seconds)
        
        self._record_request(client_id)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        })
        
        return response


def setup_middleware(app):
    """Setup all middleware for the FastAPI application."""
    
    # CORS middleware (must be first)
    app.add_middleware(
        CORSMiddleware,
        **config.cors_config
    )
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting
    if config.enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request logging (should be last to capture all processing)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware setup completed")


# Background cleanup task for rate limiting
async def cleanup_rate_limit_data():
    """Background task to clean up old rate limit data."""
    while True:
        try:
            # This would be implemented with the rate limit middleware instance
            # For now, just wait
            await asyncio.sleep(3600)  # Clean up every hour
        except Exception as e:
            logger.error(f"Error in rate limit cleanup: {e}")
            await asyncio.sleep(3600) 