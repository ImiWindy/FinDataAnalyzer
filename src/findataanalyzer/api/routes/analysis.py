"""Analysis API routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any

from findataanalyzer.core.analyzer import DataAnalyzer
from findataanalyzer.models.data_models import AnalysisRequest, AnalysisResponse
from findataanalyzer.image_analysis.tasks import process_image_task

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


@router.post("/process-image-async")
async def process_image_async(image_path: str):
    """
    Triggers an asynchronous task to process an image.

    Returns a task ID that can be used to check the status of the processing.
    """
    try:
        # Note: In a real-world app, you'd have better security and path validation.
        # This is a simplified example.
        task = process_image_task.delay(image_path)
        return {"task_id": task.id, "status": "processing_started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start task: {e}") 