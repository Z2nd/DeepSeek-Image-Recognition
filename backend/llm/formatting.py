# backend/llm/formatting.py
import json

def _format_detection_to_dict(detection_obj):
    """
    Convert a Detection object to a dictionary recursively, including sub-detections.

    Args:
        detection_obj (Detection): A single Detection object.

    Returns:
        dict: Dictionary representation of the Detection, including its features and any nested sub-detections.
    """
    detection_dict = {
        "class": detection_obj.class_name,
        "confidence": detection_obj.confidence,
        "bbox": detection_obj.bbox,
        **detection_obj.features  # Merge all key-value pairs from features
    }
    
    # Recursively format sub-detections
    if 'sub_detections' in detection_dict:
        detection_dict['sub_detections'] = [
            _format_detection_to_dict(sub_d) for sub_d in detection_dict['sub_detections']
        ]
    
    return detection_dict


def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """
    Format a list of Detection objects as a JSON string suitable for LLM input.

    Args:
        detections_list (list[Detection]): List of top-level Detection objects.
        image_shape (tuple): Shape of the image as (height, width).
        capture_time (str, optional): Timestamp of image capture. Defaults to None.

    Returns:
        str: JSON-formatted string containing detection results.
    """
    formatted_detections = [_format_detection_to_dict(d) for d in detections_list]
    
    data = {
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "capture_time": capture_time or "unknown",
        "detections": formatted_detections
    }
    
    if not detections_list:
        data["message"] = "No objects detected in the image."
        
    return json.dumps(data, indent=2)


def format_sequence_detections_for_llm(sequence_detections, capture_times):
    """
    Format detection results for a sequence of frames as a single JSON string.

    Args:
        sequence_detections (list[list[Detection]]): List of frames, each containing a list of Detection objects.
        capture_times (list[str]): List of capture times for each frame.

    Returns:
        str: JSON-formatted string summarizing all frames and their detection results.
    """
    formatted_frames = []
    for i, detections_list in enumerate(sequence_detections):
        frame_data = {
            "frame_id": i + 1,
            "capture_time": capture_times[i],
            "detections": [_format_detection_to_dict(d) for d in detections_list]
        }
        formatted_frames.append(frame_data)
    
    final_data = {
        "summary": "This is a sequence of image frames. Analyze the objects and their changes over time.",
        "frames": formatted_frames
    }
    return json.dumps(final_data, indent=2)