# backend/vision/detection.py
import cv2
import numpy as np
from ultralytics import YOLO
from .core import Detection # 导入新的Detection类

def get_initial_detections(image_bgr: np.ndarray, yolo_model: YOLO) -> list[Detection]:
    """
    仅执行初始的YOLO检测和分割，返回一个Detection对象列表。

    Args:
        image_bgr (np.ndarray): BGR格式的输入图像。
        yolo_model (YOLO): 已加载的YOLO模型。

    Returns:
        list[Detection]: 包含基础信息的Detection对象列表。
    """
    detections = []
    if yolo_model is None:
        print("Error: YOLO model is not loaded.")
        return detections
    
    img_height, img_width = image_bgr.shape[:2]
    results = yolo_model(image_bgr)

    for result in results:
        if result.boxes is None: continue
        boxes = result.boxes
        masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls = float(box.conf), int(box.cls)
            class_name = yolo_model.names[cls]
            
            mask = None
            if masks is not None and idx < len(masks.data):
                mask_np = masks.data[idx].cpu().numpy()
                mask = cv2.resize(mask_np.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            roi = image_bgr[y1:y2, x1:x2]
            
            detection_obj = Detection(
                bbox=[x1, y1, x2, y2],
                class_name=class_name,
                confidence=conf,
                mask=mask,
                roi=roi
            )
            detections.append(detection_obj)
            
    return detections


def get_combined_detections(image_bgr: np.ndarray, multi_model_config: list, model_cache: dict) -> list[Detection]:
    """
    使用多个模型进行检测，并根据配置筛选和合并结果。
    """
    from ultralytics import YOLO

    all_filtered_detections = []

    for config in multi_model_config:
        model_path = config['model_path']
        classes_to_keep = set(config['classes_to_keep']) # 使用set以提高查找效率

        # 从缓存加载或新建模型实例
        if model_path not in model_cache:
            print(f"Loading model for combination: {model_path}")
            model_cache[model_path] = YOLO(model_path)
        yolo_model = model_cache[model_path]

        if yolo_model is None:
            continue
        
        # 使用现有函数进行初步检测
        current_detections = get_initial_detections(image_bgr, yolo_model)
        
        # 根据 classes_to_keep 列表进行筛选
        filtered_detections = [
            d for d in current_detections if d.class_name in classes_to_keep
        ]
        
        all_filtered_detections.extend(filtered_detections)
        
    return all_filtered_detections