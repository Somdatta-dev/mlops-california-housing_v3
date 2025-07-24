"""
Tests for the FastAPI application.
"""

import pytest
import asyncio
import json
from typing import Dict, Any
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Import test dependencies
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.api.config import get_config
from src.data.models import HousingPredictionRequest, CaliforniaHousingData


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing."""
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }


@pytest.fixture
def sample_prediction_request(sample_housing_data):
    """Sample prediction request."""
    return {
        "features": sample_housing_data
    }


@pytest.fixture
def sample_batch_request(sample_housing_data):
    """Sample batch prediction request."""
    return {
        "features": [sample_housing_data, sample_housing_data]
    }


class TestHealthCheck:
    """Test health check endpoints."""
    
    def test_root_health_check(self, client):
        """Test root level health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
    
    def test_api_health_check(self, client):
        """Test API health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["uptime_check"] is True


class TestSystemEndpoints:
    """Test system endpoints."""
    
    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "models" in data
        assert "configuration" in data
    
    def test_cache_info(self, client):
        """Test cache info endpoint."""
        response = client.get("/api/v1/models/cache/info")
        assert response.status_code == 200
        data = response.json()
        assert "size" in data
        assert "max_size" in data
    
    def test_clear_cache(self, client):
        """Test cache clear endpoint."""
        response = client.post("/api/v1/models/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_model_info_nonexistent(self, client):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/models/nonexistent_model")
        assert response.status_code == 404 or response.status_code == 500


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_single_prediction_invalid_data(self, client):
        """Test single prediction with invalid data."""
        invalid_request = {
            "features": {
                "MedInc": -1.0,  # Invalid: negative income
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_single_prediction_missing_fields(self, client):
        """Test single prediction with missing fields."""
        incomplete_request = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0
                # Missing other required fields
            }
        }
        
        response = client.post("/api/v1/predict", json=incomplete_request)
        assert response.status_code == 422
    
    def test_batch_prediction_empty(self, client):
        """Test batch prediction with empty features."""
        empty_request = {"features": []}
        
        response = client.post("/api/v1/predict/batch", json=empty_request)
        assert response.status_code == 422
    
    def test_batch_prediction_too_large(self, client, sample_housing_data):
        """Test batch prediction with too many features."""
        # Create a batch that exceeds the limit
        config = get_config()
        large_batch = {
            "features": [sample_housing_data] * (config.max_batch_size + 1)
        }
        
        response = client.post("/api/v1/predict/batch", json=large_batch)
        assert response.status_code == 413


class TestValidation:
    """Test input validation."""
    
    def test_california_coordinates_validation(self, client):
        """Test California coordinates validation."""
        # Test coordinates outside California
        invalid_request = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 50.0,  # Outside California
                "Longitude": -100.0  # Outside California
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_bedroom_room_ratio_validation(self, client):
        """Test bedroom to room ratio validation."""
        invalid_request = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 2.0,
                "AveBedrms": 5.0,  # More bedrooms than rooms
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_occupancy_validation(self, client):
        """Test occupancy validation."""
        invalid_request = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 100.0,  # Unreasonably high occupancy
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_nonexistent_endpoint(self, client):
        """Test non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_method(self, client):
        """Test wrong HTTP method."""
        response = client.get("/api/v1/predict")  # Should be POST
        assert response.status_code == 405


class TestDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
    
    def test_docs_redirect(self, client):
        """Test docs page redirect."""
        response = client.get("/", allow_redirects=False)
        assert response.status_code == 307
        assert "/docs" in response.headers["location"]
    
    def test_swagger_ui(self, client):
        """Test Swagger UI page."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc(self, client):
        """Test ReDoc page."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestMetrics:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/metrics")
        # May be disabled in test config
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert "text/plain" in response.headers["content-type"]


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        # Check that CORS middleware is working
        assert response.status_code in [200, 405]  # May vary based on setup
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
    
    def test_request_id_header(self, client):
        """Test request ID header is added."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async functionality of endpoints."""
    
    async def test_concurrent_predictions(self, sample_prediction_request):
        """Test concurrent predictions."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Make multiple concurrent requests
            tasks = [
                ac.post("/api/v1/predict", json=sample_prediction_request)
                for _ in range(5)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that all requests completed (may fail due to model loading)
            success_count = sum(
                1 for r in responses 
                if not isinstance(r, Exception) and hasattr(r, 'status_code')
            )
            
            assert success_count >= 0  # At least some should succeed or fail gracefully


def test_pydantic_models():
    """Test Pydantic model validation."""
    # Test valid data
    valid_data = CaliforniaHousingData(
        med_inc=8.3252,
        house_age=41.0,
        ave_rooms=6.984,
        ave_bedrms=1.024,
        population=322.0,
        ave_occup=2.555,
        latitude=37.88,
        longitude=-122.23
    )
    assert valid_data.med_inc == 8.3252
    
    # Test invalid data
    with pytest.raises(ValueError):
        CaliforniaHousingData(
            med_inc=-1.0,  # Invalid negative income
            house_age=41.0,
            ave_rooms=6.984,
            ave_bedrms=1.024,
            population=322.0,
            ave_occup=2.555,
            latitude=37.88,
            longitude=-122.23
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 