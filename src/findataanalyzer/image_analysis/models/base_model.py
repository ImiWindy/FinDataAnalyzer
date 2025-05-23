"""Base model class for pattern recognition in financial charts.

This module provides the base class for all pattern recognition models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import json


class BasePatternModel(nn.Module):
    """Base class for pattern recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.is_trained = False
        self.training_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement train_step()")
    
    def validate_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of validation metrics
        """
        raise NotImplementedError("Subclasses must implement validate_step()")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            return self(x)
    
    def save(self, path: str) -> None:
        """
        Save model state to file.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model state from file.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"Model loaded from {path}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """
        Get model training history.
        
        Returns:
            List of training metrics dictionaries
        """
        return self.training_history 