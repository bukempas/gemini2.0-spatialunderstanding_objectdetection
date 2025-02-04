# Object Detection with Google Gemini 2.0 Spatial Understanding

This project demonstrates object detection in images using Google's Gemini 2.0 spatial understanding. The code identifies objects within an image, draws bounding boxes around them, and labels them with their detected names.
The example is made for detecting a weld quality inspection with Gemini 2.0 spatial understanding.

## VertexAISprint
#VertexAISprint 
"Google Cloud credits are provided for this project." 


## Overview

The script performs the following actions:

1.  **Configures the API Key:** Sets up the Google Gemini API key to authorize API calls.
2.  **Loads and Processes the Image:** Loads an image from a specified path and optionally resizes it for optimal processing by the Gemini model.
3.  **Constructs a Prompt for the Model:** Creates a detailed prompt instructing the Gemini 2.0 model to perform object detection and return the results in JSON format. The JSON structure includes the object name and bounding box coordinates (ymin, xmin, ymax, xmax) in pixel values.
4.  **Sends the Prompt and Image to the Model:** Passes the prompt and image data to the Gemini 2.0 model for processing.
5.  **Handles the Model's Response:** Receives the model's response, which is expected to be a JSON string. The code parses this string to extract the object detection information. If the response is not a valid JSON string or is empty, it prints an error message.
6.  **Draws Bounding Boxes on the Image:** If the JSON parsing is successful, the code iterates through the detected objects. For each object, it draws a red bounding box on the image using the provided coordinates and adds a text label with the object's name.
7.  **Displays the Image:** Shows the final image with bounding boxes and labels.

## Features

*   **Google Gemini 2.0 Model Integration:** Leverages the power of Google's Gemini 2.0 spatial understanding for accurate object detection.
*   **Markdown-Aware JSON Parsing:** Includes a robust `parse_json` function to handle potential markdown code block formatting (`json ... `) that the model might include in its response.
*   **Clear Error Handling:** Gracefully handles cases where the model returns invalid or incomplete JSON data.
*   **Font Customization:** Attempts to load the Arial font for clearer text labels and falls back to a default font if Arial is not found.
*   **Easy-to-Understand Code:** The code is well-commented, making it easy to follow and modify.

  ![image](https://github.com/user-attachments/assets/fd4755d4-9cc6-4356-9da6-03ad88a947b8)


  For details go to : https://github.com/google-gemini/cookbook/tree/main/gemini-2
  
  https://github.com/google-gemini/cookbook/blob/main/gemini-2/spatial_understanding.ipynb

  https://www.kaggle.com/datasets/sukmaadhiwijaya/weld-quality-inspection-instance-segmentation

  



