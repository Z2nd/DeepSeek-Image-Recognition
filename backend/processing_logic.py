import requests
import json
import traceback


def detect_objects_yolo(image_np, model):
    """
    Performs object detection using YOLO model.
    Args:
        image_np (numpy.ndarray): Image in NumPy array format (BGR).
        model (ultralytics.YOLO): Loaded YOLO model.
    Returns:
        tuple: (list of detections, annotated_image_np)
               Each detection is a dict: {'class_name': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}
               annotated_image_np is the image with detections drawn on it.
    """
    if model is None:
        print("YOLO model not loaded. Skipping detection.")
        return [], image_np

    detections_list = []
    try:
        results = model(image_np, verbose=False)  # verbose=False to reduce console output

        # Assuming results[0] contains detections for the first image
        if results and results[0]:
            annotated_image = results[0].plot() # This returns a BGR NumPy array
            boxes = results[0].boxes
            names = results[0].names # Class names

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections_list.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        else:
            annotated_image = image_np # Return original if no results
            
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return [], image_np # Return original image in case of error

    return detections_list, annotated_image

def format_detections_as_json_for_llm(detections):
    """
    Converts YOLO detections (list of dicts) into a JSON string for an LLM.
    Args:
        detections (list): List of detection dicts from detect_objects_yolo.
                           Each dict: {'class_name': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}
    Returns:
        str: A JSON string representation of the detections, or a message if no detections.
    """
    if not detections:
        return "No objects were detected in the image."
    
    # Convert the list of detection dictionaries to a JSON string
    # indent=2 makes the JSON string more readable if printed or logged.
    return json.dumps(detections, indent=2)

def generate_text_with_deepseek(prompt_text, model_name, ollama_url):
    """
    Sends a prompt to a DeepSeek model via Ollama and returns the response.
    """
    if model_name == "your-deepseek-model-name":
        return "Error: DeepSeek model name not configured. Please update 'DEEPSEEK_MODEL_NAME' in Cell 1."

    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False  # Get the full response at once
    }
    headers = {"Content-Type": "application/json"}

    print(f"\nSending prompt to DeepSeek model '{model_name}'...")
    try:
        response = requests.post(ollama_url, data=json.dumps(payload), headers=headers, timeout=180) # Increased timeout
        response.raise_for_status()
        response_data = response.json()
        if "response" in response_data:
            return response_data["response"].strip()
        else:
            return f"Error: 'response' key not found in Ollama API return. Full response: {response_data}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Ollama at {ollama_url}. Is Ollama running?"
    except requests.exceptions.Timeout:
        return f"Error: Request to Ollama timed out. The model might be taking too long or the service is unresponsive."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error from Ollama: {e}. Response: {e.response.text if e.response else 'No response body'}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON response from Ollama. Response text: {response.text}"
    except Exception as e:
        return f"Error: An unexpected error occurred while querying DeepSeek: {e}"
    
def process_image_and_describe(image_bgr, yolo_model, deepseek_model_name, ollama_api_url):
    """
    Processes a single image frame: performs YOLO detection, formats results,
    and generates a natural language description using DeepSeek via Ollama.

    Args:
        image_bgr (numpy.ndarray): The input image frame in BGR format.
        yolo_model (ultralytics.YOLO): The loaded YOLO model.
        deepseek_model_name (str): The name of the DeepSeek model in Ollama.
        ollama_api_url (str): The URL for the Ollama generate API.

    Returns:
        tuple: (natural_language_description, annotated_image_bgr)
               Returns (None, original_image_bgr) if processing fails at any step.
    """
    if image_bgr is None:
        print("Error: Input image to process_image_and_describe is None.")
        return "Error: Input image is invalid.", None

    if yolo_model is None:
        print("Error: YOLO model is not loaded.")
        return "Error: YOLO model is not available.", image_bgr.copy()

    if not deepseek_model_name or deepseek_model_name == "your-deepseek-model-name":
         print("Error: DeepSeek model name is not configured.")
         return "Error: DeepSeek model name is not configured.", image_bgr.copy()

    try:
        image_height, image_width = image_bgr.shape[:2]
        # print(f"Processing frame dimensions: Width={image_width}, Height={image_height}") # Optional: verbose

        # --- Step B: YOLO Object Detection ---
        # print("  Running YOLO Object Detection...") # Optional: verbose
        detections, annotated_image_bgr = detect_objects_yolo(image_bgr.copy(), yolo_model)

        # --- Step C: Formatting Detections as JSON for LLM ---
        # print("  Formatting Detections as JSON...") # Optional: verbose
        detections_json_string = format_detections_as_json_for_llm(detections)
        # print(f"  Detections JSON:\n{detections_json_string}") # Optional: verbose

        # --- Step D: DeepSeek Text Generation ---
        # print("  Generating Text with DeepSeek...") # Optional: verbose

        if detections_json_string == "No objects were detected in the image.":
            prompt_for_llm = f"""
You are an AI assistant that describes images.
The object detection process found no specific objects in an image that is {image_width} pixels wide and {image_height} pixels high.
Please provide a very brief, general description acknowledging this, for example, "The image appears to have no distinct objects detected."
"""
        else:
            prompt_for_llm = f"""
You are an AI assistant that describes images based on a list of detected objects provided in JSON format.
The image dimensions are {image_width} pixels wide and {image_height} pixels high.
The following is a JSON list of objects detected in the image. Each object has:
- 'class_name': The identified type of the object.
- 'confidence': The model's confidence in this detection (a value between 0.0 and 1.0).
- 'bbox': The bounding box coordinates [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the object within the image dimensions mentioned above.
Detected objects data:
{detections_json_string}
Based on this structured data, please generate a concise and natural language description of the scene. Try to infer relationships between objects and their general locations (e.g., "left side", "center", "behind another object") rather than just listing the raw bounding box coordinates. Focus on creating a human-like, coherent description of what the image likely contains.
"""
        # print(f"  Prompt being sent to DeepSeek:\n{prompt_for_llm}") # Optional: verbose

        natural_language_description = generate_text_with_deepseek(prompt_for_llm, deepseek_model_name, ollama_api_url)

        # --- Step E: Output Natural Language Description (Handled by caller) ---
        # The description is returned, the caller will print it.

        return natural_language_description, annotated_image_bgr

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        traceback.print_exc()
        return f"Error processing image: {e}", image_bgr.copy() # Return original image on error