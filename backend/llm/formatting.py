# backend/llm/formatting.py
import json

def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """将单帧的检测结果格式化为JSON字符串。"""
    data = {
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "capture_time": capture_time or "unknown",
        "detections": [
            {
                "class": d["class"], "group": d["group"], "confidence": d["confidence"],
                "bbox": d["bbox"], "color_name": d["color_name"], "mask_area": d["mask_area"],
                "sub_class": d["sub_class"], "sub_confidence": d["sub_confidence"]
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
                    "class": d["class"], "group": d["group"], "confidence": d["confidence"],
                    "bbox": d["bbox"], "color_name": d.get("color_name", "unknown"),
                    "mask_area": d.get("mask_area", 0.0), "sub_class": d.get("sub_class", "unknown"),
                    "is_moving": d.get("is_moving", False)
                } for d in detections_list
            ]
        }
        formatted_frames.append(frame_data)
    
    final_data = {
        "summary": "This is a sequence of image frames. Analyze the objects and their changes over time.",
        "frames": formatted_frames
    }
    return json.dumps(final_data, indent=2)