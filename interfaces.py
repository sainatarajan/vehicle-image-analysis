from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image


class VehicleDetector(ABC):
    """Abstract base class for vehicle detection implementations."""
    
    @abstractmethod
    async def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the given image.
        
        Args:
            image: The input image as a numpy array.
            
        Returns:
            A list of detected vehicles with their bounding boxes, 
            confidence scores, and class labels.
            Example:
            [
                {
                    'label': 'car',
                    'confidence': 0.95,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'image_section': np.ndarray  # cropped vehicle image
                }
            ]
        """
        pass
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the vehicle detection model."""
        pass


class CaptionGenerator(ABC):
    """Abstract base class for image caption generation implementations."""
    
    @abstractmethod
    async def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image: The input image as a numpy array.
            
        Returns:
            A string caption describing the image content.
        """
        pass
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the caption generation model."""
        pass


class ImageAnalyzer(ABC):
    """Abstract base class for image analyzers that orchestrate detection and captioning."""
    
    @abstractmethod
    async def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze the given image to detect vehicles and generate a caption.
        
        Args:
            image: The input image as a PIL Image object.
            
        Returns:
            A dictionary containing the analysis results.
            Example:
            {
                'total_cars': 5,
                'red_cars': 2,
                'description': 'A group of cars parked on a street.'
            }
        """
        pass
