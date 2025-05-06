import numpy as np
import os
import cv2

from typing import Dict, Any, List
from PIL import Image
from loguru import logger

from interfaces import ImageAnalyzer, VehicleDetector, CaptionGenerator
from utils import is_red_vehicle


class DefaultImageAnalyzer(ImageAnalyzer):
    """An implementation of ImageAnalyzer that orchestrates detection and captioning."""
    
    def __init__(self, vehicle_detector: VehicleDetector, caption_generator: CaptionGenerator):
        """
        Initialize the DefaultImageAnalyzer with detector and captioner.
        
        Args:
            vehicle_detector: An implementation of VehicleDetector.
            caption_generator: An implementation of CaptionGenerator.
        """
        self.vehicle_detector = vehicle_detector
        self.caption_generator = caption_generator
        logger.info("DefaultImageAnalyzer initialized")
    
    async def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze the given image to detect vehicles and generate a caption.
        
        Args:
            image: The input image as a PIL Image object.
            
        Returns:
            A dictionary containing the analysis results with total_cars, red_cars, and description.
        """
        try:
            logger.info("Starting image analysis")
            
            # Convert PIL Image to numpy array. PIL images are RGB by default
            image_np = np.array(image)
            
            # If the image has an alpha channel, remove it
            if image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
                
            # IMPORTANT: Convert RGB (PIL default) to BGR (OpenCV expected format)
            image_np_bgr = image_np[:, :, ::-1].copy()
            
            # Save debug image to verify correct colors
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/input_image_bgr.jpg", image_np_bgr)
            
            logger.info(f"Image converted from RGB (PIL) to BGR (OpenCV) format, shape={image_np_bgr.shape}")
            
            # Detect vehicles using the BGR image (OpenCV format)
            vehicles = await self.vehicle_detector.detect_vehicles(image_np_bgr)
            total_cars = len(vehicles)
            
            # Count red cars
            red_cars = 0
            for vehicle in vehicles:
                if is_red_vehicle(vehicle["image_section"]):
                    red_cars += 1
            
            # Generate image caption using the BGR image
            description = await self.caption_generator.generate_caption(image_np_bgr)
            
            logger.info(f"Analysis complete: {total_cars} cars detected, {red_cars} red cars")
            
            # Return the analysis results
            return {
                "total_cars": total_cars,
                "red_cars": red_cars,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise