# backend/llm/formatting.py (修正版)
import json

def _format_detection_to_dict(detection_obj):
    """
    辅助函数：将一个Detection对象及其特征转换为字典，并递归处理子检测。
    """
    # --- 修正：使用点符号(.)访问对象属性 ---
    detection_dict = {
        "class": detection_obj.class_name,
        "confidence": detection_obj.confidence,
        "bbox": detection_obj.bbox,
        **detection_obj.features  # 将features字典中的所有键值对解包并合并
    }
    
    # 递归处理子检测
    if 'sub_detections' in detection_dict:
        # detection_dict['sub_detections'] 是一个 Detection 对象的列表
        detection_dict['sub_detections'] = [
            _format_detection_to_dict(sub_d) for sub_d in detection_dict['sub_detections']
        ]
    
    return detection_dict


def format_detections_as_json_for_llm(detections_list, image_shape, capture_time=None):
    """将检测结果列表（可能包含嵌套）格式化为JSON字符串。"""
    
    # 此处的 d 是一个字典，而不是 Detection 对象，因此调用 _format_detection_to_dict 
    # 的是 detection 对象列表，而非字典列表
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
    """将帧序列的检测结果格式化为单个JSON字符串。"""
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