import cv2
import numpy as np
import json
import requests
from ultralytics import YOLO
from sklearn.cluster import KMeans
import time
import psutil
import re

def get_dominant_color(image, mask=None, color_space='HSV', n_clusters=1):
    """
    Extract dominant color from an image region using K-Means in specified color space.
    Args:
        image: BGR image (NumPy array)
        mask: Binary mask (optional, same size as image)
        color_space: 'HSV' or 'HSL' (default: 'HSV')
        n_clusters: Number of clusters for K-Means (default: 1)
    Returns:
        dominant_color: List of [H, S, V] or [H, S, L] values
        color_name: Approximate color name (e.g., 'red', 'blue')
    """
    # Color name mapping for Hue (simplified, adjust as needed)
    color_names = {
        (0, 15): 'red', (15, 45): 'orange', (45, 75): 'yellow',
        (75, 165): 'green', (165, 195): 'cyan', (195, 255): 'blue',
        (255, 345): 'purple', (345, 360): 'red'
    }

    # Extract region of interest (ROI)
    roi = image[mask > 0] if mask is not None else image.reshape(-1, 3)
    if len(roi) == 0:
        return [0, 0, 0], 'unknown'

    # Convert to specified color space
    if color_space == 'HSV':
        roi_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[mask > 0] if mask is not None else cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    elif color_space == 'HSL':
        roi_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[mask > 0] if mask is not None else cv2.cvtColor(image, cv2.COLOR_BGR2HLS).reshape(-1, 3)
    else:
        raise ValueError("Unsupported color space. Use 'HSV' or 'HSL'.")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(roi_converted)
    dominant_color = kmeans.cluster_centers_[0].astype(float).tolist()  # [H, S, V] or [H, S, L]

    # Normalize Hue to 0-360 for HSV/HSL (OpenCV uses 0-180)
    dominant_color[0] *= 2  # Convert OpenCV Hue (0-180) to 0-360

    # Determine approximate color name based on Hue
    hue = dominant_color[0]
    color_name = 'unknown'
    for (h_min, h_max), name in color_names.items():
        if h_min <= hue <= h_max:
            color_name = name
            break

    # Adjust saturation and value/lightness for meaningful color name
    if color_space == 'HSV' and (dominant_color[1] < 0.2 or dominant_color[2] < 0.2):
        color_name = 'gray' if dominant_color[2] < 0.5 else 'white'
    elif color_space == 'HSL' and (dominant_color[1] < 0.2 or abs(dominant_color[2] - 0.5) > 0.4):
        color_name = 'gray' if dominant_color[2] < 0.5 else 'white'

    return dominant_color, color_name

def detect_objects_yolo(image_bgr, yolo_model, color_space='HSV'):
    """
    Modified YOLO detection with HSV/HSL color analysis.
    Args:
        image_bgr: NumPy image array in BGR format
        yolo_model: Loaded YOLO segmentation model
        color_space: 'HSV' or 'HSL' (default: 'HSV')
    Returns:
        detections_list: List of detection results with dominant color and color name
        annotated_image: Annotated image with bounding boxes and labels
    """
    try:
        if yolo_model is None:
            print("Error: YOLO model is not loaded.")
            return [], image_bgr

        results = yolo_model(image_bgr)
        detections_list = []
        annotated_image = image_bgr.copy()
        img_height, img_width = image_bgr.shape[:2]

        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = yolo_model.names[cls]

                mask_area = 0.0
                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks.data[idx].cpu().numpy()
                    mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    mask_area = np.sum(mask) / (img_width * img_height)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, (255, 0, 0), 2)

                roi = image_bgr * mask[..., None] if mask is not None else image_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Compute color histogram (optional, omitted for simplicity)
                # hist_r = cv2.calcHist([roi], [2], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                # ...

                # Get dominant color in HSV or HSL
                dominant_color, color_name = get_dominant_color(roi, mask, color_space=color_space)

                detections_list.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "dominant_color": dominant_color,
                    "color_name": color_name,
                    "mask_area": float(mask_area)
                })

                # Draw bounding box and label
                label = f"{class_name} ({color_name}) {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections_list, annotated_image

    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return [], image_bgr

def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """
    Format detection results into a JSON string, including dominant color RGB, mask area, histogram, and capture time.
    Args:
        detections_list: List of detection results
        image_shape: Image dimensions (height, width, channels)
        capture_time: Capture time string (optional)
    Returns:
        JSON string
    """
    if not detections_list:
        data = {
            "image_height": image_shape[0],
            "image_width": image_shape[1],
            "detections": [],
            "message": "No objects detected in the image."
        }
    else:
        formatted_detections = []
        for detection in detections_list:
            formatted_detections.append({
                "class": detection["class"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "dominant_color": detection["dominant_color"],
                "color_name": detection["color_name"],
                "mask_area": detection["mask_area"],
                # "color_histogram": detection["color_histogram"]
            })
        data = {
            "image_height": image_shape[0],
            "image_width": image_shape[1],
            "detections": formatted_detections
        }
    
    if capture_time:
        data["capture_time"] = capture_time
    
    return json.dumps(data)
        
def answer_question_with_deepseek(json_detections, question, ollama_api_url, model_name, max_retries=3, retry_delay=2):
    """
    Generate answers using DeepSeek and collect performance metrics (inference time, memory, CPU usage).
    Args:
        json_detections: JSON string of detection results
        question: User input question
        ollama_api_url: Ollama API URL
        model_name: DeepSeek model name
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries in seconds
    Returns:
        tuple: (answer text or error message, performance metrics dictionary)
    """
    try:
        # Initialize performance metrics
        metrics = {
            "question": question,
            "start_time": time.time(),
            "inference_time": 0.0,
            "memory_mb": 0.0,
            "cpu_percent": 0.0,
            "retry_attempts": 0,
            "status": "success"
        }
        
        # Get process for memory and CPU monitoring
        process = psutil.Process()
        
        detections = json.loads(json_detections)
        image_height = detections["image_height"]
        image_width = detections["image_width"]
        capture_time = detections.get("capture_time", "unknown")

        if not detections["detections"]:
            prompt = (
                f"The image is {image_height} pixels high and {image_width} pixels wide. "
                f"No objects were detected. The image was captured at {capture_time}. "
                f"Please answer the following question based on this information:\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                "You are an AI assistant that answers questions about an image based on structured detection data from a computer vision model. "
                "'dominant_color_rgb' is the RGB color value [R, G, B] (0-255) of the dominant color in the object's mask. "
                "The arguements of the bounding box 'bbox' are in the format [x_center, y_center, width, height]"
                f"The image is {image_height} pixels high and {image_width} pixels wide. "
                f"The image was captured at {capture_time}. "
                f"Detected objects data:\n{json.dumps(detections, indent=2)}\n"
                "Based on this data, answer the following question in concise, natural language. "
                f"Question: {question}"
            )

        # Retry mechanism for API call
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(ollama_api_url, json=payload, timeout=1200)
                response.raise_for_status()
                complete_response = response.json().get("response", "No answer generated.")

                # Extract final answer (remove <think>...</think> tags)
                final_answer = re.sub(r'<think>.*?</think>', '', complete_response, flags=re.DOTALL).strip()
                if not final_answer:
                    final_answer = complete_response  # Fallback if no content outside <think> tags
                
                metrics["inference_time"] = time.time() - start_time
                metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024  # Convert to MB
                metrics["retry_attempts"] = attempt
                return final_answer, complete_response, metrics
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                metrics["retry_attempts"] = attempt + 1
                print(f"API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        metrics["status"] = "failed"
        metrics["error"] = "Failed to get response from Ollama API"
        return "Error: Failed to get response from Ollama API after {max_retries} attempts.", "", metrics

    except json.JSONDecodeError:
        metrics["status"] = "failed"
        metrics["error"] = "Invalid response format from Ollama API"
        return "Error: Invalid response format from Ollama API.", "", metrics
    except Exception as e:
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        return f"Error in answering question: {str(e)}", "", metrics

def process_image_and_describe(image_bgr, yolo_model, model_name, ollama_api_url, capture_time=None):
    """
    Process the image and return detection results and annotated image for subsequent questioning and histogram display.
    Args:
        image_bgr: NumPy image array in BGR format
        yolo_model: Loaded YOLO segmentation model
        model_name: DeepSeek model name
        ollama_api_url: Ollama API URL
        capture_time: Capture time string (optional)
    Returns:
        json_detections: JSON string of detection results (including time, mask area, histogram, and RGB)
        annotated_image: Annotated image (BGR format)
    """
    try:
        # Perform object detection and segmentation
        detections_list, annotated_image = detect_objects_yolo(image_bgr, yolo_model)

        # Format detection results
        json_detections = format_detections_as_json_for_llm(detections_list, image_bgr.shape, capture_time)

        return json_detections, annotated_image

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return json.dumps({"message": f"Processing failed: {str(e)}"}), image_bgr