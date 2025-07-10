import cv2
import numpy as np
import json
import requests
from ultralytics import YOLO
from sklearn.cluster import KMeans
import time

def detect_objects_yolo(image_bgr, yolo_model):
    """
    Perform instance segmentation using the YOLO model, extracting class, confidence, bounding box, mask area, color histogram, and dominant color RGB via K-Means.
    Args:
        image_bgr: NumPy image array in BGR format
        yolo_model: Loaded YOLO segmentation model
    Returns:
        detections_list: List of detection results, including class, confidence, bounding box, mask area, histogram, and dominant color RGB
        annotated_image: Annotated image (BGR format, with bounding boxes, mask contours, and dominant color)
    """
    try:
        if yolo_model is None:
            print("Error: YOLO model is not loaded.")
            return [], image_bgr

        # Perform object detection and segmentation
        results = yolo_model(image_bgr)
        detections_list = []
        annotated_image = image_bgr.copy()
        img_height, img_width = image_bgr.shape[:2]

        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for idx, box in enumerate(boxes):
                # Extract bounding box coordinates, class, and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = yolo_model.names[cls]

                # Extract mask (if available)
                mask_area = 0.0
                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks.data[idx].cpu().numpy() # Mask as boolean array
                    # Resize mask to match input image dimensions
                    mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    mask_area = np.sum(mask) / (image_bgr.shape[0] * image_bgr.shape[1])  # Area proportion
                    # Draw mask contours
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, (255, 0, 0), 2)

                # Extract mask or bounding box region
                roi = image_bgr * mask[..., None] if mask is not None else image_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Compute color histogram (RGB, 8 bins/channel)
                hist_r = cv2.calcHist([roi], [2], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                hist_g = cv2.calcHist([roi], [1], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                hist_b = cv2.calcHist([roi], [0], mask.astype(np.uint8) if mask is not None else None, [8], [0, 256])
                # Normalize histogram
                hist_r = hist_r.flatten().tolist()
                hist_g = hist_g.flatten().tolist()
                hist_b = hist_b.flatten().tolist()
                hist_sum = sum(hist_r) + sum(hist_g) + sum(hist_b)
                if hist_sum > 0:
                    hist_r = [x / hist_sum for x in hist_r]
                    hist_g = [x / hist_sum for x in hist_g]
                    hist_b = [x / hist_sum for x in hist_b]

                # Extract dominant color using K-Means (RGB)
                pixels = roi[mask > 0] if mask is not None else roi.reshape(-1, 3)
                if len(pixels) > 0:
                    kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
                    kmeans.fit(pixels)
                    dominant_color_rgb = kmeans.cluster_centers_[0].astype(int).tolist()  # [R, G, B]
                else:
                    dominant_color_rgb = [0, 0, 0]  # Default value

                # Add to detection results
                detections_list.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "dominant_color_rgb": dominant_color_rgb,
                    "mask_area": float(mask_area),
                    "color_histogram": {
                        "r": hist_r,
                        "g": hist_g,
                        "b": hist_b
                    }
                })

                # Draw bounding box and label (including RGB value)
                label = f"{class_name} (RGB: {dominant_color_rgb}) {conf:.2f}"
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
                "dominant_color_rgb": detection["dominant_color_rgb"],
                # "mask_area": detection["mask_area"],
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
    Generate answers using DeepSeek based on detection results, mask area, histogram, dominant color RGB, capture time, and user question.
    Args:
        json_detections: JSON string of detection results
        question: User input question
        ollama_api_url: Ollama API URL
        model_name: DeepSeek model name
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries in seconds
    Returns:
        Answer text or error message
    """
    try:
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
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(ollama_api_url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("response", "No answer generated.")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        return f"Error: Failed to get response from Ollama API after {max_retries} attempts."

    except json.JSONDecodeError:
        return "Error: Invalid response format from Ollama API."
    except Exception as e:
        return f"Error in answering question: {str(e)}"

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