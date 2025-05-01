import numpy as np
import os
import cv2

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import Dict, List, Tuple, Any
import logging
from loguru import logger

from interfaces import VehicleDetector
from config import settings


class DetrVehicleDetector(VehicleDetector):
    """Implementation of VehicleDetector using DETR (DEtection TRansformer) model."""
    
    def __init__(self, model_name: str = settings.vehicle_detector_model,
                 confidence_threshold: float = settings.vehicle_detection_confidence_threshold,
                 vehicle_class: str = settings.vehicle_class_label,
                 device: str = settings.device):
        """
        Initialize the DETR vehicle detector.
        
        Args:
            model_name: The name or path of the DETR model.
            confidence_threshold: The minimum confidence score for a detection to be considered valid.
            vehicle_class: The class label for vehicles to detect.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.vehicle_class = vehicle_class
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.processor = None
        logger.info(f"Initializing DETR Vehicle Detector with model: {model_name} on {self.device}")
        
    async def load_model(self) -> None:
        """Load the DETR model and processor."""
        try:
            logger.info(f"Loading DETR model: {self.model_name}")
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"DETR model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load DETR model: {str(e)}")
            raise
    
    async def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the given image using DETR.
        
        Args:
            image: The input image as a numpy array (BGR format from OpenCV).
            
        Returns:
            A list of detected vehicles with their bounding boxes, 
            confidence scores, and class labels.
        """
        if self.model is None or self.processor is None:
            logger.error("Model not loaded. Please call load_model() first.")
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            # Convert BGR to RGB for the DETR model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save debug image to verify correct conversion
            import os
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/detector_input_bgr.jpg", image)
            cv2.imwrite("debug/detector_converted_rgb.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Converted BGR to RGB for vehicle detection, shape={image_rgb.shape}")
                
            # Convert to PIL Image for the processor
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Process image for DETR
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert outputs to COCO API
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]
            
            # Extract vehicle detections
            vehicles = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()]
                
                # Only consider vehicle class
                if label_name == self.vehicle_class:
                    x_min, y_min, x_max, y_max = box.tolist()
                    
                    # Convert to integers for cropping
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    
                    # Ensure the coordinates are within the image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(pil_image.width, x_max)
                    y_max = min(pil_image.height, y_max)
                    
                    # Crop the vehicle from the original image
                    vehicle_image = image[y_min:y_max, x_min:x_max]
                    
                    vehicles.append({
                        'label': label_name,
                        'confidence': score.item(),
                        'bbox': [x_min, y_min, x_max, y_max],
                        'image_section': vehicle_image
                    })
            
            logger.info(f"Detected {len(vehicles)} vehicles in the image")
            return vehicles
            
        except Exception as e:
            logger.error(f"Error during vehicle detection: {str(e)}")
            raise