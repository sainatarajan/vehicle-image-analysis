import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from loguru import logger

from config import settings


def save_debug_visualization(vehicle_image: np.ndarray, red_mask: np.ndarray, vehicle_id: int = 0):
    """
    Save debug visualization of the red detection process.
    
    Args:
        vehicle_image: Original vehicle image
        red_mask: Mask of detected red pixels
        vehicle_id: ID to distinguish multiple vehicles in one image
    """
    try:
        import os
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save original vehicle crop
        cv2.imwrite(f"{debug_dir}/vehicle_{vehicle_id}_original.jpg", vehicle_image)
        
        # Save the red mask
        cv2.imwrite(f"{debug_dir}/vehicle_{vehicle_id}_red_mask.jpg", red_mask)
        
        # Create a visualization with the mask overlaid
        mask_overlay = vehicle_image.copy()
        mask_overlay[red_mask > 0] = [0, 0, 255]  # Highlight red areas in blue
        cv2.imwrite(f"{debug_dir}/vehicle_{vehicle_id}_overlay.jpg", mask_overlay)
        
        logger.info(f"Debug visualization saved to {debug_dir}/vehicle_{vehicle_id}_*.jpg")
    except Exception as e:
        logger.error(f"Error saving debug visualization: {str(e)}")


def is_red_vehicle(
    vehicle_image: np.ndarray,
    hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = settings.red_hsv_ranges,
    threshold_percentage: float = settings.red_pixel_threshold_percentage
) -> bool:
    """
    Determine if a vehicle is predominantly red based on HSV color thresholds.
    
    Args:
        vehicle_image: Cropped image of the vehicle as a numpy array (BGR format).
        hsv_ranges: List of HSV color ranges that define "red".
        threshold_percentage: Minimum percentage of red pixels required to classify as a red vehicle.
    
    Returns:
        True if the vehicle is red, False otherwise.
    """
    try:
        # Check if the image is valid
        if vehicle_image is None or vehicle_image.size == 0:
            logger.warning("Empty vehicle image received for red detection")
            return False
        
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for red pixels using the HSV ranges
        red_mask = np.zeros(vehicle_image.shape[:2], dtype=np.uint8)
        
        for lower, upper in hsv_ranges:
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            current_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            red_mask = cv2.bitwise_or(red_mask, current_mask)
        
        # Calculate the percentage of red pixels
        total_pixels = vehicle_image.shape[0] * vehicle_image.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
        
        # Determine if the vehicle is red based on the threshold
        is_red = red_percentage >= threshold_percentage
        
        # Increase log level for red detection to make it more visible
        logger.info(f"Red pixel percentage: {red_percentage:.4f}, Threshold: {threshold_percentage:.2f}, Is red: {is_red}")
        
        # Save debug visualization
        save_debug_visualization(vehicle_image, red_mask)
        
        return is_red
        
    except Exception as e:
        logger.error(f"Error during red vehicle detection: {str(e)}")
        return False


async def setup_logger(log_level: str = settings.log_level) -> None:
    """
    Configure the logger.
    
    Args:
        log_level: The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger.remove()  # Remove default handler
    logger.add(
        "logs/api.log",
        rotation="500 MB",
        retention="10 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        lambda msg: print(msg),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        colorize=True,
    )
    logger.info(f"Logger configured with level: {log_level}")