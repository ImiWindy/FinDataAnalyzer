"""Integration tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
import tempfile
import pandas as pd
from pathlib import Path
import json

from findataanalyzer.api.main import app
from findataanalyzer.models.data_models import AnalysisRequest


class TestAPI:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to FinDataAnalyzer API"}
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_analyze_endpoint(self, client, sample_ohlc_data):
        """Test the analyze endpoint."""
        # Save sample data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            sample_ohlc_data.to_csv(tmp_path, index=False)
        
        try:
            # Create analysis request
            request = AnalysisRequest(
                data_source=tmp_path,
                parameters={"advanced": True}
            )
            
            # Send POST request to analyze endpoint
            response = client.post(
                "/api/v1/analysis/analyze",
                json=request.dict()
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            
            assert result["success"] is True
            assert "results" in result
            assert "summary_stats" in result["results"]
            assert "time_series" in result["results"]
            assert "correlation" in result["results"]
            assert "advanced" in result["results"]
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_upload_analyze_endpoint(self, client, sample_ohlc_data):
        """Test the upload and analyze endpoint."""
        # Prepare data as CSV content
        csv_content = sample_ohlc_data.to_csv(index=False)
        
        # Create file-like object
        files = {
            "file": ("test_data.csv", csv_content, "text/csv")
        }
        
        # Send POST request to upload-analyze endpoint
        response = client.post(
            "/api/v1/analysis/upload-analyze",
            files=files
        )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        assert "filename" in result
        assert result["filename"] == "test_data.csv"
        assert "results" in result
        assert "summary_stats" in result["results"]
        assert "time_series" in result["results"]
        assert "correlation" in result["results"]
    
    def test_analyze_invalid_data_source(self, client):
        """Test analyze endpoint with an invalid data source."""
        # Create request with non-existent data source
        request = AnalysisRequest(
            data_source="non_existent_file.csv"
        )
        
        # Send POST request
        response = client.post(
            "/api/v1/analysis/analyze",
            json=request.dict()
        )
        
        # Verify response indicates error
        assert response.status_code == 500
        assert "detail" in response.json() 