"""Analysis API routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any

from findataanalyzer.core.analyzer import DataAnalyzer
from findataanalyzer.models.data_models import AnalysisRequest, AnalysisResponse

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Analyze financial data based on request parameters."""
    try:
        analyzer = DataAnalyzer()
        results = analyzer.analyze(request.data_source, request.parameters)
        return AnalysisResponse(
            success=True,
            results=results,
            message="Analysis completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a file and analyze its contents."""
    try:
        contents = await file.read()
        analyzer = DataAnalyzer()
        results = analyzer.analyze_raw_data(contents)
        return {
            "filename": file.filename,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 