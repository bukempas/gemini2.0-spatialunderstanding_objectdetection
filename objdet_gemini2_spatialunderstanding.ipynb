# Import necessary libraries
from google.colab import userdata
import os
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import json

# Configure API Key and Model with Google Colab
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
model_name = "gemini-2.0-flash-exp"

# Function to parse JSON from a string
def parse_json(json_output):
    """Parses JSON from a string, removing potential markdown code block formatting."""
    lines = json_output.splitlines()
    json_content = ""
    in_json_block = False

    for line in lines:
        if line.strip() == "```json":
            in_json_block = True
        elif line.strip() == "```" and in_json_block:
            in_json_block = False  # End of JSON block
        elif in_json_block:
            json_content += line + "\n"

    return json_content.strip()

# Load and preprocess the image
image_path = "/content/your-file.jpg"  # Ensure this path is correct
img = Image.open(image_path)
max_size = 300
if img.width > max_size or img.height > max_size:
    img.thumbnail((max_size, max_size))
width, height = img.size

# Craft the prompt
prompt = """You are an expert of welding quality. Detect the defects, if there are any, in the welding image and provide bounding box information in JSON format. The output should be a single, complete JSON string and nothing else. Do not include any text other than the JSON string.

The JSON output should include:

*   object_name: The name of the detected object.
*   bounding_box: A list containing the coordinates [ymin, xmin, ymax, xmax] of the bounding box just on the object in pixel values.

Example JSON structure:

[
    {
        "object_name": "cat",
        "bounding_box": [10, 20, 150, 200]
    },
    {
        "object_name": "dog",
        "bounding_box": [200, 100, 350, 300]
    }
]
"""

# Generate content and handle response
response = client.models.generate_content(
    model=model_name,
    contents=[prompt, img]
)
raw_response = response.text

# Parse the JSON response
parsed_json_string = parse_json(raw_response)

if not parsed_json_string:
    print("Model returned an empty or incomplete JSON response.")
    print(f"Raw response text: {raw_response}")
    spatial_info = []
else:
    try:
        spatial_info = json.loads(parsed_json_string)
        print(spatial_info)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response text: {raw_response}")
        print(f"Parsed JSON string: {parsed_json_string}")
        spatial_info = []

# Draw bounding boxes if JSON parsing was successful
draw = ImageDraw.Draw(img)
for obj in spatial_info:
    object_name = obj["object_name"]
    if obj["bounding_box"]:
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=5)
        draw.text((xmin, ymin - 10), object_name, fill="red")

# Save or show the image with bounding boxes
img.show()
img.save("/content/annotated_image.jpg")
