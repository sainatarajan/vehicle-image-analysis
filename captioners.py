import numpy as np
import os
import cv2
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from typing import List, Dict, Any
from loguru import logger

from interfaces import CaptionGenerator
from config import settings


class TransformerCaptionGenerator(CaptionGenerator):
    """Implementation of CaptionGenerator using a transformer-based model."""
    
    def __init__(self, model_name: str = settings.caption_generator_model,
                 device: str = settings.device):
        """
        Initialize the transformer-based caption generator.
        
        Args:
            model_name: The name or path of the caption generation model.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        logger.info(f"Initializing Transformer Caption Generator with model: {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load the captioning model, feature extractor, and tokenizer."""
        try:
            logger.info(f"Loading caption generation model: {self.model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Caption generation model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load caption generation model: {str(e)}")
            raise
    
    async def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image: The input image as a numpy array (BGR format from OpenCV).
            
        Returns:
            A string caption describing the image content.
        """
        if self.model is None or self.feature_extractor is None or self.tokenizer is None:
            logger.error("Model not loaded. Please call load_model() first.")
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            # Convert BGR to RGB for the captioning model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save debug image to verify correct conversion
            import os
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/captioner_input_bgr.jpg", image)
            cv2.imwrite("debug/captioner_converted_rgb.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Converted BGR to RGB for caption generation, shape={image_rgb.shape}")
            
            # Convert to PIL Image for the feature extractor
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Extract features from the image
            pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values, 
                    max_length=50, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            # Decode the caption
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error during caption generation: {str(e)}")
            raise