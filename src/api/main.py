"""
Main FastAPI application for California Housing Price Prediction API.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

from ..utils.logging_config import setup_logging, get_logger
from .config import setup_environment, get_config
from .middleware import setup_middleware
from .endpoints import router
from .model_loader import initialize_model_loader
from .metrics import start_metrics_collection, get_metrics
from .exceptions import APIException

# Setup configuration and logging
config = setup_environment()
setup_logging(**config.logging_config)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting California Housing Price Prediction API")
    logger.info(f"Version: {config.app_version}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"GPU monitoring: {config.enable_gpu_monitoring}")
    logger.info(f"Prometheus metrics: {config.enable_prometheus}")
    
    try:
        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = await initialize_model_loader()
        
        # Preload default model
        logger.info(f"Preloading default model: {config.default_model_name}")
        try:
            await model_loader.get_default_model()
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload default model: {e}")
        
        # Start metrics collection
        if config.enable_prometheus:
            logger.info("Starting metrics collection...")
            await start_metrics_collection()
        
        # Log available models
        available_models = model_loader.get_available_models()
        logger.info(f"Found {len(available_models)} available models")
        for model in available_models:
            logger.info(f"  - {model['name']} v{model['version']} ({model['algorithm']})")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down California Housing Price Prediction API")
    
    try:
        # Clear model cache
        model_loader = await initialize_model_loader()
        model_loader.cache.clear()
        logger.info("Model cache cleared")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("API shutdown completed")


def create_custom_openapi():
    """Create custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=config.app_name,
        version=config.app_version,
        description=config.app_description,
        routes=app.routes,
    )
    
    # Add custom information
    openapi_schema["info"]["contact"] = {
        "name": "MLOps Team",
        "email": "mlops@example.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://{config.host}:{config.port}",
            "description": "Development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": config.api_key_header
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI application
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description=config.app_description,
    lifespan=lifespan,
    docs_url=None,  # Disable default docs to use custom
    redoc_url=None,  # Disable default redoc to use custom
    openapi_url="/openapi.json"
)

# Set custom OpenAPI
app.openapi = create_custom_openapi

# Setup middleware
setup_middleware(app)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Global exception handler for API exceptions."""
    logger.error(f"API Exception: {exc.detail} (Code: {exc.error_code})")
    
    return {
        "error": exc.detail,
        "error_code": exc.error_code,
        "request_id": getattr(request.state, 'request_id', None)
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Additional request logging middleware."""
    # This is handled by RequestLoggingMiddleware, but we can add extra logic here if needed
    response = await call_next(request)
    return response


# Add some additional routes for development
if config.debug:
    @app.get("/debug/config", include_in_schema=False)
    async def debug_config():
        """Debug endpoint to view configuration (only in debug mode)."""
        return {
            "app_name": config.app_name,
            "version": config.app_version,
            "debug": config.debug,
            "default_model": config.default_model_name,
            "model_cache_size": config.model_cache_size,
            "max_batch_size": config.max_batch_size,
            "cors_origins": config.cors_origins,
            "enable_prometheus": config.enable_prometheus,
            "enable_gpu_monitoring": config.enable_gpu_monitoring,
            "enable_rate_limiting": config.enable_rate_limiting
        }
    
    @app.get("/debug/routes", include_in_schema=False)
    async def debug_routes():
        """Debug endpoint to view all routes (only in debug mode)."""
        routes = []
        for route in app.routes:
            if hasattr(route, 'methods'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name
                })
        return {"routes": routes}


# Health check at root level for load balancers
@app.get("/health", tags=["System"])
async def root_health_check():
    """Root level health check for load balancers."""
    return {"status": "healthy", "service": config.app_name}


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.lower(),
        workers=config.workers if not config.reload else 1
    ) 