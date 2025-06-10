"""Celery tasks for image analysis."""

import logging
from typing import Union
from pathlib import Path

from findataanalyzer.core.celery_app import celery_app
from findataanalyzer.image_analysis.processors.image_processor import StandardProcessor

logger = logging.getLogger(__name__)

@celery_app.task(name="tasks.process_image")
def process_image_task(image_path_str: str, grayscale: bool = False, normalize: bool = True):
    """
    A Celery task to process a financial chart image.
    
    Args:
        image_path_str: The string path to the input image.
        grayscale: Whether to convert the image to grayscale.
        normalize: Whether to normalize pixel values.
    """
    try:
        logger.info("Celery task started for image: %s", image_path_str)
        image_path = Path(image_path_str)
        
        if not image_path.exists():
            logger.error("Image path does not exist: %s", image_path)
            return {"status": "error", "message": "Image not found"}

        # Initialize the processor
        processor = StandardProcessor()
        
        # Process the image
        processed_path = processor.process(
            image_path, 
            grayscale=grayscale, 
            normalize=normalize
        )
        
        logger.info("Image processing complete. Output at: %s", processed_path)
        return {"status": "success", "processed_path": processed_path}
        
    except Exception as e:
        logger.exception("Error during image processing task for %s", image_path_str)
        return {"status": "error", "message": str(e)} 