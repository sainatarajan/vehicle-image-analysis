from pydantic_settings import BaseSettings
from typing import Dict, List, Tuple


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    api_title: str = "Car Analysis API"
    api_description: str = "API for analyzing images and detecting red cars"
    api_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model settings
    vehicle_detector_model: str = "facebook/detr-resnet-50"
    caption_generator_model: str = "nlpconnect/vit-gpt2-image-captioning"
    
    # Detection settings
    vehicle_detection_confidence_threshold: float = 0.7
    vehicle_class_label: str = "car"
    
    # Red color detection settings. Expanded ranges to catch more shades of red
    red_hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
        # Primary red range (lower saturation threshold to catch brighter reds)
        ((0, 70, 50), (10, 255, 255)),
        # Secondary red range (wraps around the hue spectrum)
        ((160, 70, 50), (179, 255, 255))
    ]
    # Lowered threshold percentage to be more lenient
    red_pixel_threshold_percentage: float = 0.15
    
    # Performance settings
    device: str = "cuda"  # "cpu" or "cuda"
    
    # Logging settings
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings()