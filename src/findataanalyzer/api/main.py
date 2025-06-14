"""FastAPI main application for FinDataAnalyzer."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from findataanalyzer.api.routes import analysis, models
from findataanalyzer.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for financial data analysis",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(analysis.router, prefix=settings.API_V1_STR)
app.include_router(models.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to FinDataAnalyzer API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 