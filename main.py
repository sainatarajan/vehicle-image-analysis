import io
import asyncio
from typing import Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from config import settings
from utils import setup_logger
from interfaces import VehicleDetector, CaptionGenerator, ImageAnalyzer
from detectors import DetrVehicleDetector
from captioners import TransformerCaptionGenerator
from analyzers import DefaultImageAnalyzer


# Global variables for model instances
vehicle_detector: VehicleDetector = None
caption_generator: CaptionGenerator = None
image_analyzer: ImageAnalyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI lifespan that handles model loading and unloading.
    
    This is executed on startup and shutdown of the application.
    """
    # Setup logging
    await setup_logger(settings.log_level)
    
    # Load models on startup
    logger.info("Initializing models...")
    try:
        global vehicle_detector, caption_generator, image_analyzer
        
        # Initialize and load vehicle detector
        vehicle_detector = DetrVehicleDetector(
            model_name=settings.vehicle_detector_model,
            confidence_threshold=settings.vehicle_detection_confidence_threshold,
            vehicle_class=settings.vehicle_class_label,
            device=settings.device
        )
        await vehicle_detector.load_model()
        
        # Initialize and load caption generator
        caption_generator = TransformerCaptionGenerator(
            model_name=settings.caption_generator_model,
            device=settings.device
        )
        await caption_generator.load_model()
        
        # Initialize image analyzer
        image_analyzer = DefaultImageAnalyzer(vehicle_detector, caption_generator)
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down, releasing resources...")


# Create the FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def validate_image(file: UploadFile) -> Image.Image:
    """
    Validate and process the uploaded image file.
    
    Args:
        file: The uploaded file.
        
    Returns:
        A PIL Image object.
        
    Raises:
        HTTPException: If the file is not a valid image.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="Uploaded file is not an image. Please upload an image file."
            )
        
        # Read the file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400, 
                detail="Empty file. Please upload a valid image."
            )
        
        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(content))
            return image
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail="Failed to process the image. Please upload a valid image file."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing the image."
        )


@app.post("/analyze-image", response_model=Dict[str, Any])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image to detect cars and generate a caption.
    
    Args:
        file: The image file to analyze.
        
    Returns:
        A JSON object containing:
        - total_cars: Total number of cars detected.
        - red_cars: Number of red cars detected.
        - description: A text description of the image content.
    """
    try:
        # Ensure models are loaded
        if not vehicle_detector or not caption_generator or not image_analyzer:
            raise HTTPException(
                status_code=503, 
                detail="Models are not initialized. Please try again later."
            )
        
        # Validate and process the image
        image = await validate_image(file)
        
        # Analyze the image
        logger.info(f"Analyzing image: {file.filename}")
        results = await image_analyzer.analyze(image)
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while analyzing the image."
        )


def start():
    """Entry point for running the application."""
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    start()
