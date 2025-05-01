import requests
import os
import json
import sys

def analyze_image(image_path, api_url="http://localhost:8000/analyze-image"):
    """
    Send an image to the API for analysis.
    
    Args:
        image_path: Path to the image file.
        api_url: URL of the API endpoint.
        
    Returns:
        The JSON response from the API.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        # Open the image file
        with open(image_path, 'rb') as image_file:
            # Create the files dict for the request
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            
            # Send the POST request to the API
            print(f"Sending {image_path} to {api_url}...")
            response = requests.post(api_url, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                print("Analysis successful!")
                print(f"Total cars: {result['total_cars']}")
                print(f"Red cars: {result['red_cars']}")
                print(f"Description: {result['description']}")
                return result
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
    
    except Exception as e:
        print(f"Error sending request: {str(e)}")
        return None

if __name__ == "__main__":
    # Check if image path was provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to the first image in the images directory
        images_dir = "./images"
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            sys.exit(1)
            
        image_path = os.path.join(images_dir, image_files[0])
        print(f"No image specified, using first image found: {image_path}")
    
    # Analyze the image
    result = analyze_image(image_path)
    
    # Pretty print the full result
    if result:
        print("\nFull JSON response:")
        print(json.dumps(result, indent=2))
