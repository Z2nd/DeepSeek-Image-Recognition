# backend/vision/detection.py
import cv2
import numpy as np
from ultralytics import YOLO

from backend import config
from backend.vision.features import get_dominant_color
from backend.vision.classification import secondary_classification
# 导入新的绘图模块
from backend.vision.drawing import draw_enhanced_annotations

def detect_objects_yolo(image_bgr, yolo_model, secondary_model=None, color_space='HSV', motion_mask=None):
    """
    使用YOLO模型进行实例分割，并整合所有特征分析。
    (注意：此版本不再直接绘图，而是将绘图任务交给 draw_enhanced_annotations)
    """
    try:
        if yolo_model is None:
            print("Error: YOLO model is not loaded.")
            return [], image_bgr

        results = yolo_model(image_bgr)
        detections_list = []
        img_height, img_width = image_bgr.shape[:2]

        for result in results:
            if result.boxes is None: continue
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls = float(box.conf), int(box.cls)
                class_name = yolo_model.names[cls]
                group = next((g for g, ids in config.CATEGORY_GROUPS.items() if cls in ids), 'other')

                mask, roi, mask_area = None, image_bgr[y1:y2, x1:x2], 0.0
                if masks is not None and idx < len(masks.data):
                    mask_np = masks.data[idx].cpu().numpy()
                    mask = cv2.resize(mask_np.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    mask_area = np.sum(mask > 0) / (img_width * img_height)
                
                if roi.size == 0: continue

                # 运动检测
                is_moving = False
                if motion_mask is not None and mask is not None:
                    intersection = cv2.bitwise_and(mask, motion_mask)
                    if np.sum(mask > 0) > 0 and (np.sum(intersection > 0) / np.sum(mask > 0)) > 0.2:
                        is_moving = True

                # 主导颜色
                dominant_color, color_name = get_dominant_color(image_bgr, mask=mask, color_space=color_space)

                # 二次分类
                sub_class, sub_conf = 'unknown', 0.0
                if secondary_model and group in ['animal', 'vehicle']:
                    class_map = config.ANIMAL_SUBCLASSES if group == 'animal' else config.VEHICLE_SUBCLASSES
                    class_names = class_map.get(class_name, [])
                    sub_class, sub_conf = secondary_classification(roi, secondary_model, class_names)

                detections_list.append({
                    "class": class_name, "group": group, "confidence": conf,
                    "bbox": [x1, y1, x2, y2], "dominant_color": dominant_color,
                    "color_name": color_name, "mask_area": float(mask_area),
                    "sub_class": sub_class, "sub_confidence": sub_conf, "is_moving": is_moving
                })

        # --- 绘图逻辑分离 ---
        # 使用新的增强型绘图函数来创建带标注的图像
        # 掩码轮廓的绘制依然在这里，因为它依赖于检测循环中的'mask'变量
        annotated_image = image_bgr.copy()
        if masks:
            for mask_data in masks:
                mask_np = mask_data.data[0].cpu().numpy()
                mask_img = cv2.resize(mask_np.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_image, contours, -1, (255, 0, 0), 2)
        
        # 调用新函数绘制边界框和标签
        annotated_image = draw_enhanced_annotations(annotated_image, detections_list)
        
        return detections_list, annotated_image
    
    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return [], image_bgr

def detect_motion_with_optical_flow(prev_frame_gray, current_frame_gray, threshold=2.0):
    """使用光流法计算运动掩码。"""
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = np.zeros_like(prev_frame_gray)
    motion_mask[magnitude > threshold] = 255
    return motion_mask