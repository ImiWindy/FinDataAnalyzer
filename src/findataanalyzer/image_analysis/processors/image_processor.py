"""Image processor module for financial chart images.

This module provides functionality for preprocessing and augmenting financial chart images.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os
from abc import ABC, abstractmethod


class ImageProcessor(ABC):
    """Base class for image processors."""
    
    def __init__(self, input_dir: str = "data/raw/charts", output_dir: str = "data/processed/charts"):
        """Initialize an image processor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def process(self, image_path: Union[str, Path], **kwargs) -> str:
        """Process a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the processed image
        """
        pass
    
    def process_batch(self, image_paths: List[Union[str, Path]], **kwargs) -> List[str]:
        """Process a batch of images.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of paths to processed images
        """
        processed_paths = []
        for image_path in image_paths:
            try:
                processed_path = self.process(image_path, **kwargs)
                processed_paths.append(processed_path)
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
        
        return processed_paths
    
    def save_image(self, image: np.ndarray, filename: str) -> str:
        """Save an image to a file.
        
        Args:
            image: Image data as NumPy array
            filename: Name of the file to save
            
        Returns:
            Path to the saved image
        """
        file_path = self.output_dir / filename
        cv2.imwrite(str(file_path), image)
        
        self.logger.info(f"Saved processed image to {file_path}")
        return str(file_path)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from a file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image data as NumPy array
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image


class StandardProcessor(ImageProcessor):
    """Standard image processor for financial charts."""
    
    def __init__(self, input_dir: str = "data/raw/charts", output_dir: str = "data/processed/charts/standard",
                 target_size: Tuple[int, int] = (224, 224)):
        """Initialize a standard image processor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            target_size: Target size (width, height) for resizing images
        """
        super().__init__(input_dir, output_dir)
        self.target_size = target_size
    
    def process(self, image_path: Union[str, Path], **kwargs) -> str:
        """Process a single image with standard preprocessing.
        
        The standard preprocessing includes:
        - Resizing to target size
        - Converting to grayscale (optional)
        - Normalizing pixel values
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the processed image
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load the image
        image = self.load_image(image_path)
        
        # Get options from kwargs
        grayscale = kwargs.get("grayscale", False)
        normalize = kwargs.get("normalize", True)
        
        # Resize the image
        image = cv2.resize(image, self.target_size)
        
        # Convert to grayscale if requested
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Add channel dimension back for consistency
            image = image[:, :, np.newaxis]
        
        # Normalize pixel values if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
            # Convert back to uint8 for saving
            image = (image * 255).astype(np.uint8)
        
        # Generate output filename
        input_path = Path(image_path)
        output_filename = f"processed_{input_path.name}"
        
        # Save the processed image
        return self.save_image(image, output_filename)


class ChartAugmentor(ImageProcessor):
    """Image augmentor for financial charts."""
    
    def __init__(self, input_dir: str = "data/processed/charts/standard", 
                 output_dir: str = "data/processed/charts/augmented"):
        """Initialize a chart augmentor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save augmented images
        """
        super().__init__(input_dir, output_dir)
    
    def process(self, image_path: Union[str, Path], **kwargs) -> str:
        """Process a single image with augmentation.
        
        The augmentation includes:
        - Random brightness and contrast adjustments
        - Random rotation (small angles)
        - Random zooming
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the augmented image
        """
        self.logger.info(f"Augmenting image: {image_path}")
        
        # Load the image
        image = self.load_image(image_path)
        
        # Get augmentation parameters from kwargs or use random values
        brightness = kwargs.get("brightness", np.random.uniform(0.8, 1.2))
        contrast = kwargs.get("contrast", np.random.uniform(0.8, 1.2))
        angle = kwargs.get("angle", np.random.uniform(-5, 5))
        zoom = kwargs.get("zoom", np.random.uniform(0.9, 1.1))
        
        # Apply brightness and contrast adjustments
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        # Apply rotation
        if angle != 0:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Apply zoom
        if zoom != 1:
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # Crop the center of the image if zooming in
            if zoom > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = cv2.resize(image, (new_w, new_h))
                image = image[start_h:start_h+h, start_w:start_w+w]
            # Pad the image if zooming out
            else:
                resized = cv2.resize(image, (new_w, new_h))
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = np.zeros((h, w, 3), dtype=np.uint8)
                image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Generate output filename
        input_path = Path(image_path)
        augmentation_desc = f"b{brightness:.1f}_c{contrast:.1f}_a{angle:.1f}_z{zoom:.1f}"
        output_filename = f"aug_{augmentation_desc}_{input_path.name}"
        
        # Save the augmented image
        return self.save_image(image, output_filename)
    
    def augment_multiple(self, image_path: Union[str, Path], num_augmentations: int = 5) -> List[str]:
        """Create multiple augmented versions of an image.
        
        Args:
            image_path: Path to the input image
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of paths to augmented images
        """
        augmented_paths = []
        for i in range(num_augmentations):
            try:
                # Generate random augmentation parameters
                params = {
                    "brightness": np.random.uniform(0.8, 1.2),
                    "contrast": np.random.uniform(0.8, 1.2),
                    "angle": np.random.uniform(-5, 5),
                    "zoom": np.random.uniform(0.9, 1.1)
                }
                
                # Process the image with these parameters
                augmented_path = self.process(image_path, **params)
                augmented_paths.append(augmented_path)
            except Exception as e:
                self.logger.error(f"Error augmenting image {image_path}: {e}")
        
        return augmented_paths


class PatternExtractor(ImageProcessor):
    """Extractor for chart patterns from financial chart images."""
    
    def __init__(self, input_dir: str = "data/processed/charts/standard", 
                 output_dir: str = "data/processed/charts/patterns"):
        """Initialize a pattern extractor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save pattern images
        """
        super().__init__(input_dir, output_dir)
    
    def process(self, image_path: Union[str, Path], **kwargs) -> str:
        """Process a single image to extract patterns.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the processed image with highlighted patterns
        """
        self.logger.info(f"Extracting patterns from image: {image_path}")
        
        # Load the image
        image = self.load_image(image_path)
        
        # Get options from kwargs
        pattern_type = kwargs.get("pattern_type", "support_resistance")
        
        # This would be a complex implementation that uses computer vision techniques
        # to identify chart patterns. For now, we'll just return a placeholder.
        self.logger.info(f"Would extract {pattern_type} patterns from {image_path}")
        
        # Generate output filename
        input_path = Path(image_path)
        output_filename = f"pattern_{pattern_type}_{input_path.name}"
        
        # In a real implementation, we would detect patterns and highlight them
        # For now, just return the original image
        return self.save_image(image, output_filename) 