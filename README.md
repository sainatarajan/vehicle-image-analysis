# Image Analysis API

A RESTful API built with Python and FastAPI for analyzing images, specifically designed to detect red cars, and generate descriptive captions.

## Features

- **Vehicle Detection**: Uses DETR (DEtection TRansformer) model to detect cars in images
- **Color Analysis**: Counts the number of red cars using HSV color space analysis
- **Image Captioning**: Generates descriptive captions for images using a ViT-GPT2 model
- **Scalable Architecture**: Designed with OOP and SOLID principles for maintainability and scalability
- **Comprehensive Logging**: Detailed logs and debug visualizations for troubleshooting

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

```bash
# Clone the repository
git clone https://github.com/sainatarajan/vehicle-image-analysis.git
cd image-analysis-api

# Install dependencies with Poetry
poetry install

# Verify CUDA installation (optional)
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the API

### Start the server

```bash
# Using Poetry
poetry run python main.py

# Using pip (with activated virtual environment)
python main.py
```

The server will start on http://0.0.0.0:8000 by default.

## Using the API

### Endpoint: `/analyze-image`

- **Method**: POST
- **Content-Type**: multipart/form-data
- **Form Key**: `file` (an image file)



### Sample Request using Python

The repository includes a `send_request.py` script that can be used to easily test the API:

```bash
# Install requests library if needed
pip install requests

# Run with a specific image
python send_request.py ./images/1.jpg

# Or run without arguments to analyze the first image in the ./images directory
python send_request.py
```

The script will output:
1. The API request details
2. A summary of results (total cars, red cars, description)
3. The complete JSON response

Example output:
```
Sending ./images/1.jpg to http://localhost:8000/analyze-image...
Analysis successful!
Total cars: 1
Red cars: 1
Description: A red sports car parked on a white background.

Full JSON response:
{
  "total_cars": 1,
  "red_cars": 1,
  "description": "A red sports car parked on a white background."
}
```

### Sample Response

```json
{
  "total_cars": 1,
  "red_cars": 1,
  "description": "A red sports car parked on a white background."
}
```

## Example Results

Below is a visual demonstration of how the API processes an image of a car:

### Input Car Image
![Input Car Image](images/1.jpg)

### Red Car Mask (HSV Color Detection)
![Red Car Mask](debug/vehicle_0_red_mask.jpg)

### API Response (JSON)
```json
{
  "total_cars": 1,
  "red_cars": 1,
  "description": "A shiny red sports car with alloy wheels."
}
```

## Architecture

The project follows SOLID principles and is organized into several modules:

- **main.py**: FastAPI application setup and endpoint definitions
- **interfaces.py**: Abstract base classes for components
- **detectors.py**: Vehicle detection implementation using DETR
- **captioners.py**: Image caption generation implementation
- **analyzers.py**: Orchestrator class combining detection and captioning
- **utils.py**: Helper functions, including red car detection
- **config.py**: Application settings using pydantic-settings

### Color Detection

The system uses HSV (Hue, Saturation, Value) color space for detecting red cars, which provides several advantages over RGB:

- Better separation of color from brightness and saturation
- More intuitive color ranges
- Better handling of lighting variations
- Easier detection of specific colors like red

## Debugging

When processing images, the API generates debug visualizations in the `debug` directory:

- Original vehicle crops
- Color masks showing detected red areas
- Color conversion verification images

These debug images can help troubleshoot issues with color detection or model performance.



## License

[MIT License](LICENSE)
