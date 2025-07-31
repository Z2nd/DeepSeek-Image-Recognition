# backend/llm/formatting.py
import json

def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """将单帧的检测结果格式化为JSON字符串。"""

    features_copy = detections_list[0].get("features", {}).copy()

    data = {
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "capture_time": capture_time or "unknown",
        "detections": [
            {
                "class": d["class_name"], "confidence": d["confidence"],
                "bbox": d["bbox"], 
                **{k: v for k, v in features_copy.items() if k != "dominant_color_hsv"},
            } for d in detections_list
        ]
    }
    if not detections_list:
        data["message"] = "No objects detected in the image."
    return json.dumps(data, indent=2)

def format_sequence_detections_for_llm(sequence_detections, capture_times):
    """将帧序列的检测结果格式化为单个JSON字符串。"""
    formatted_frames = []
    for i, detections_list in enumerate(sequence_detections):
        frame_data = {
            "frame_id": i + 1,
            "capture_time": capture_times[i],
            "detections": [
                {
                    "class": d["class"], "confidence": d["confidence"],
                    "bbox": d["bbox"], 
                    **d.get("features", {}),
                } for d in detections_list
            ]
        }
        formatted_frames.append(frame_data)
    
    final_data = {
        "summary": "This is a sequence of image frames. Analyze the objects and their changes over time.",
        "frames": formatted_frames
    }
    return json.dumps(final_data, indent=2)