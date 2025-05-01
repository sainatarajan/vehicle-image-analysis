# Image Analysis API

A RESTful API built with Python and FastAPI for analyzing images, specifically designed to detect vehicles, identify car colors, and generate descriptive captions.

## Features

- **Vehicle Detection**: Uses DETR (DEtection TRansformer) model to identify cars in images
- **Color Analysis**: Detects red vehicles using HSV color space analysis
- **Image Captioning**: Generates descriptive captions for images using a transformer-based model
- **Scalable Architecture**: Designed with OOP and SOLID principles for maintainability and scalability
- **Comprehensive Logging**: Detailed logs and debug visualizations for troubleshooting

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/image-analysis-api.git
cd image-analysis-api

# Install dependencies with Poetry
poetry install

# Verify CUDA installation (optional)
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/image-analysis-api.git
cd image-analysis-api

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
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

### Sample Request using curl

```bash
curl -X POST \
  http://localhost:8000/analyze-image \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./images/your_car_image.jpg"
```

### Sample Request using Python

```python
import requests

def analyze_image(image_path, api_url="http://localhost:8000/analyze-image"):
    with open(image_path, 'rb') as image_file:
        files = {'file': (image_file.name, image_file, 'image/jpeg')}
        response = requests.post(api_url, files=files)
        
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

result = analyze_image("./images/car.jpg")
print(result)
```

### Sample Response

```json
{
  "total_cars": 2,
  "red_cars": 1,
  "description": "A red car and a blue car parked on the street."
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

## Scaling for Production

For high-load production environments, consider:

1. Running multiple worker processes with Gunicorn:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

2. Using optimized inference servers like TorchServe or Triton Inference Server

3. Implementing non-blocking inference patterns using asyncio.to_thread

4. Model optimization techniques like quantization or TensorRT/ONNX conversion

## License

[MIT License](LICENSE)
