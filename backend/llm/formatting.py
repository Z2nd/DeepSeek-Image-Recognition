# backend/llm/formatting.py
import json

def _format_detection_to_dict(detection_obj):
    """
    Helper function: Convert a Detection object and its attributes to a dictionary, recursively processing sub-detections.
    """
    # --- Fix: Use dot notation to access object attributes ---
    detection_dict = {
        "class": detection_obj.class_name,
        "confidence": detection_obj.confidence,
        "bbox": detection_obj.bbox,
        **detection_obj.features  # Unpack and merge all key-value pairs from the features dictionary
    }
    
    # Recursively process sub-detections
    if 'sub_detections' in detection_dict:
        # detection_dict['sub_detections'] 是一个 Detection 对象的列表
        detection_dict['sub_detections'] = [
            _format_detection_to_dict(sub_d) for sub_d in detection_dict['sub_detections']
        ]
    
    return detection_dict


def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """Format a list of detection results (possibly nested) as a JSON string."""
    
    # Here, d is a Detection object, not a dictionary, so _format_detection_to_dict
    # is called on a list of Detection objects, not a list of dictionaries.
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
    """Format the detection results of a sequence of frames as a single JSON string."""
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