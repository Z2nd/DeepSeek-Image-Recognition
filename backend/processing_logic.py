import cv2
import numpy as np
import json
import requests
from ultralytics import YOLO
from sklearn.cluster import KMeans
import time
import psutil
import re
import torch
from torchvision import models, transforms

# Definition COCO80 Category Grouping
CATEGORY_GROUPS = {
    'person': [0],  # person
    'animal': [14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 41, 44],  # bird, dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe, fish, snake
    'vehicle': [2, 3, 4, 5, 6, 7, 8, 9],  # bicycle, car, motorcycle, airplane, bus, train, truck, boat
    'other': [i for i in range(80) if i not in [14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 41, 44, 2, 3, 4, 5, 6, 7, 8, 9]]
}

# Animal subcategory mapping (example, to be adapted to the dataset)
ANIMAL_SUBCLASSES = {
    'dog': ['golden retriever', 'labrador', 'husky', 'bulldog'],
    'cat': ['persian', 'siamese', 'maine coon']
}

# Transportation subcategory mapping (example, to be adapted to the dataset)
VEHICLE_SUBCLASSES = {
    'car': ['sedan', 'suv', 'pickup'],
    'bus': ['city bus', 'school bus']
}

def load_secondary_model(model_type='mobilenet_v3_small', num_classes=10):
    """
    Load a pre-trained secondary model for classification (e.g., MobileNetV3-Small).
    Args:
        model_type: Model type ('mobilenet_v3_small' or others)
        num_classes: Number of output classes for classification
    Returns:
        model: Loaded PyTorch model in evaluation mode
    """
    try:
        if model_type == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading secondary model: {str(e)}")
        return None

def preprocess_roi_for_classification(roi, target_size=(224, 224)):
    """
    Preprocess ROI for secondary classification.
    Args:
        roi: BGR image (NumPy array)
        target_size: Target size for resizing (width, height)
    Returns:
        tensor: Preprocessed image tensor for PyTorch
    """
    try:
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize to target size
        roi_resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_AREA)
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        roi_tensor = transform(roi_resized).unsqueeze(0)  # Add batch dimension
        return roi_tensor
    except Exception as e:
        print(f"Error preprocessing ROI: {str(e)}")
        return None

def secondary_classification(roi, model, class_names, group):
    """
    Perform secondary classification on ROI to predict sub-class.
    Args:
        roi: BGR image (NumPy array)
        model: Loaded secondary classification model
        class_names: List of sub-class names (e.g., ['golden retriever', 'labrador'])
        group: Category group ('animal' or 'vehicle')
    Returns:
        sub_class: Predicted sub-class name
        confidence: Classification confidence
    """
    try:
        if model is None or roi is None:
            return 'unknown', 0.0
        
        # Preprocess ROI
        roi_tensor = preprocess_roi_for_classification(roi)
        if roi_tensor is None:
            return 'unknown', 0.0
        
        # Perform inference
        with torch.no_grad():
            outputs = model(roi_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            sub_class = class_names[predicted_idx.item()]
        
        return sub_class, float(confidence)
    except Exception as e:
        print(f"Error in secondary classification: {str(e)}")
        return 'unknown', 0.0

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
    color_names = {
        (0, 15): 'red', (15, 45): 'orange', (45, 75): 'yellow',
        (75, 165): 'green', (165, 195): 'cyan', (195, 255): 'blue',
        (255, 345): 'purple', (345, 360): 'red'
    }

    roi = image[mask > 0] if mask is not None else image.reshape(-1, 3)
    if len(roi) == 0:
        return [0, 0, 0], 'unknown'

    if color_space == 'HSV':
        roi_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[mask > 0] if mask is not None else cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    elif color_space == 'HSL':
        roi_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[mask > 0] if mask is not None else cv2.cvtColor(image, cv2.COLOR_BGR2HLS).reshape(-1, 3)
    else:
        raise ValueError("Unsupported color space. Use 'HSV' or 'HSL'.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(roi_converted)
    dominant_color = kmeans.cluster_centers_[0].astype(float).tolist()
    dominant_color[0] *= 2  # Convert OpenCV Hue (0-180) to 0-360

    hue = dominant_color[0]
    color_name = 'unknown'
    for (h_min, h_max), name in color_names.items():
        if h_min <= hue <= h_max:
            color_name = name
            break

    if color_space == 'HSV' and (dominant_color[1] < 0.2 or dominant_color[2] < 0.2):
        color_name = 'gray' if dominant_color[2] < 0.5 else 'white'
    elif color_space == 'HSL' and (dominant_color[1] < 0.2 or abs(dominant_color[2] - 0.5) > 0.4):
        color_name = 'gray' if dominant_color[2] < 0.5 else 'white'

    return dominant_color, color_name

def detect_objects_yolo(image_bgr, yolo_model, secondary_model=None, color_space='HSV', motion_mask=None):
    """
    Perform instance segmentation using YOLO model with grouping and secondary classification.
    Args:
        image_bgr: NumPy image array in BGR format
        yolo_model: Loaded YOLO segmentation model
        secondary_model: Loaded secondary classification model (optional)
        color_space: 'HSV' or 'HSL' (default: 'HSV')
    Returns:
        detections_list: List of detection results with group and sub-class
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

                # Assign group based on COCO80 class ID
                group = next((g for g, ids in CATEGORY_GROUPS.items() if cls in ids), 'other')

                # Extract mask and ROI
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

                # --- Determining whether an object is in motion ---
                is_moving = False
                if motion_mask is not None and mask is not None:
                    # Compute the intersection of the object mask and the motion mask
                    intersection = cv2.bitwise_and(mask, motion_mask)
                    intersection_area = np.sum(intersection > 0)
                    object_area = np.sum(mask > 0)
                    
                    # If the proportion of the intersection area to the total area of the object exceeds a threshold (e.g., 20%), 
                    # the object is considered to be in motion
                    if object_area > 0 and (intersection_area / object_area) > 0.2:
                        is_moving = True

                # Get dominant color
                dominant_color, color_name = get_dominant_color(roi, mask, color_space=color_space)

                # Perform secondary classification (if applicable)
                sub_class, sub_conf = 'unknown', 0.0
                if secondary_model and group in ['animal', 'vehicle']:
                    class_names = ANIMAL_SUBCLASSES.get(class_name, []) if group == 'animal' else VEHICLE_SUBCLASSES.get(class_name, [])
                    if class_names:
                        sub_class, sub_conf = secondary_classification(roi, secondary_model, class_names, group)

                detections_list.append({
                    "class": class_name,
                    "group": group,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "dominant_color": dominant_color, 
                    "color_name": color_name,         
                    "mask_area": float(np.sum(mask > 0) / (img_width * img_height)) if mask is not None else 0.0,
                    "sub_class": sub_class,           
                    "sub_confidence": sub_conf,       
                    "is_moving": is_moving            
                })

                # Draw bounding box and label
                label = f"{class_name} ({sub_class}, {color_name}) {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections_list, annotated_image

    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return [], image_bgr

def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """
    Format detection results into a JSON string, including group and sub-class.
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
                "group": detection["group"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "dominant_color": detection["dominant_color"],
                "color_name": detection["color_name"],
                "mask_area": detection["mask_area"],
                "sub_class": detection["sub_class"],
                "sub_confidence": detection["sub_confidence"]
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
    Generate answers using DeepSeek with updated prompt including group and sub-class.
    Args:
        json_detections: JSON string of detection results
        question: User input question
        ollama_api_url: Ollama API URL
        model_name: DeepSeek model name
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries in seconds
    Returns:
        tuple: (answer text, complete response, performance metrics dictionary)
    """
    metrics = {
        "question": question,
        "start_time": time.time(),
        "inference_time": 0.0,
        "memory_mb": 0.0,
        "retry_attempts": 0,
        "status": "pending"
    }
    process = psutil.Process()

    try:
        detections_data = json.loads(json_detections)

        # --- Determine if it's a single graph JSON or a sequence JSON ---
        if "frames" in detections_data:
            # Processing Frame Sequences
            prompt = (
                "You are an AI assistant analyzing a sequence of image frames. "
                "Analyze the objects and their changes over time based on the following structured data. "
                "Each item in the 'frames' list represents one frame with a frame_id and its detected objects.\n"
                f"Data: {json.dumps(detections_data, indent=2)}\n\n"
                "Please answer the following user question in concise, natural descriptive language, one sentence:\n"
                f"Question: {question}"
            )
        else:
            # Processing Single Graph
            image_height = detections_data.get("image_height", "unknown")
            image_width = detections_data.get("image_width", "unknown")
            capture_time = detections_data.get("capture_time", "unknown")

            if not detections_data.get("detections"):
                prompt = (
                    f"The image is {image_height} pixels high and {image_width} pixels wide. "
                    f"No objects were detected. The image was captured at {capture_time}. "
                    f"Please answer the following question based on this information:\n"
                    f"Question: {question}"
                )
            else:
                prompt = (
                    "You are an AI assistant that answers questions about an image based on structured detection data. "
                    f"The image is {image_height} pixels high and {image_width} pixels wide, captured at {capture_time}. "
                    "Detected objects data (bbox: [x1, y1, x2, y2], dominant_color: [H, S, V], color_name: common name, "
                    "group: category group, sub_class: fine-grained class):\n"
                    f"{json.dumps(detections_data, indent=2)}\n"
                    "Answer the following question in concise, natural language:\n"
                    f"Question: {question}"
                )

        # API Calls
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                payload = {"model": model_name, "prompt": prompt, "stream": False}
                response = requests.post(ollama_api_url, json=payload, timeout=1200)
                response.raise_for_status()
                complete_response = response.json().get("response", "No answer generated.")
                
                final_answer = re.sub(r'<think>.*?</think>', '', complete_response, flags=re.DOTALL).strip()
                if not final_answer:
                    final_answer = complete_response
                
                metrics["inference_time"] = time.time() - start_time
                metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
                metrics["retry_attempts"] = attempt
                metrics["status"] = "success"
                return final_answer, complete_response, metrics
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                metrics["retry_attempts"] = attempt + 1
                print(f"API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        metrics["status"] = "failed"
        metrics["error"] = f"Failed to get response from Ollama API after {max_retries} attempts."
        return metrics["error"], "", metrics

    except Exception as e:
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        return f"Error in answering question: {str(e)}", "", metrics

def process_image_and_describe(image_bgr, yolo_model, model_name, ollama_api_url, capture_time=None, secondary_model=None):
    """
    Process the image with YOLO and secondary classification, return detection results and annotated image.
    Args:
        image_bgr: NumPy image array in BGR format
        yolo_model: Loaded YOLO segmentation model
        model_name: DeepSeek model name
        ollama_api_url: Ollama API URL
        capture_time: Capture time string (optional)
        secondary_model: Loaded secondary classification model (optional)
    Returns:
        json_detections: JSON string of detection results
        annotated_image: Annotated image with bounding boxes and masks
    """
    try:
        detections_list, annotated_image = detect_objects_yolo(image_bgr, yolo_model, secondary_model=secondary_model)
        json_detections = format_detections_as_json_for_llm(detections_list, image_bgr.shape, capture_time)
        return json_detections, annotated_image
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return json.dumps({"message": f"Processing failed: {str(e)}"}), image_bgr
    
def format_sequence_detections_for_llm(sequence_detections, capture_times):
    """
    Format the detection results of the frame sequence into a single JSON string for LLM.

    Args:
        sequence_detections: A list of detection results, where each element is a single-frame detections_list.
        capture_times: A list of capture times, corresponding to sequence_detections.

    Returns:
        JSON string
    """
    formatted_frames = []
    for i, detections_list in enumerate(sequence_detections):
        if not detections_list:
            formatted_detections = []
        else:
            formatted_detections = [
                {
                    "class": detection["class"],
                    "group": detection["group"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "color_name": detection.get("color_name", "unknown"),
                    "mask_area": detection.get("mask_area", 0.0),
                    "sub_class": detection.get("sub_class", "unknown"),
                    "is_moving": detection.get("is_moving", False) 
                } for detection in detections_list
            ]
        
        frame_data = {
            "frame_id": i + 1,
            "capture_time": capture_times[i],
            "detections": formatted_detections
        }
        formatted_frames.append(frame_data)
    
    # Adjust the prompt to the LLM to inform it that it is processing a sequence
    final_data = {
        "summary": "This is a sequence of image frames. Analyze the objects and their changes over time.",
        "frames": formatted_frames
    }
    
    return json.dumps(final_data, indent=2)

def detect_motion_with_optical_flow(prev_frame_gray, current_frame_gray, threshold=2.0):
    """
    The dense optical flow between two frames is computed using the Farneback algorithm and a motion mask is returned.

    Args:
        prev_frame_gray (numpy.ndarray): The grayscale image of the previous frame.
        current_frame_gray (numpy.ndarray): The grayscale image of the current frame.
        threshold (float): Threshold for the motion amplitude to determine if the pixel is moving.

    Returns:
        numpy.ndarray: A binarized motion mask with a white (255) motion region.
    """
    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray, 
        current_frame_gray, 
        None, 
        0.5,  # pyr_scale: Pyramid scaling
        3,    # levels: Pyramid levels
        15,   # winsize: Average window size
        3,    # iterations: Number of times per iteration
        5,    # poly_n: Pixel Neighborhood Size
        1.2,  # poly_sigma: Gaussian standard deviation (statistics)
        0     # flags
    )
    
    # Calculate the magnitude and angle of the motion vector for each pixel
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create a motion mask to keep only pixels with motion greater than a threshold value
    motion_mask = np.zeros_like(prev_frame_gray)
    motion_mask[magnitude > threshold] = 255
    
    return motion_mask
