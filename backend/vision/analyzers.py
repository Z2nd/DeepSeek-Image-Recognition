# backend/vision/analyzers.py
import numpy as np
from ultralytics import YOLO
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color

class GroupingAnalyzer(Analyzer):
    """根据类别ID为物体分组。"""
    def analyze(self, detection: Detection, **kwargs):
        yolo_model = kwargs.get('yolo_model')
        if not yolo_model: return
        
        cls_id = next((k for k, v in yolo_model.names.items() if v == detection.class_name), None)
        if cls_id is not None:
            group = next((g for g, ids in config.CATEGORY_GROUPS.items() if cls_id in ids), 'other')
            detection.add_feature('group', group)
        else:
            detection.add_feature('group', 'unknown')

class ColorAnalyzer(Analyzer):
    """分析物体的主导颜色。"""
    def analyze(self, detection: Detection, **kwargs):
        original_image = kwargs.get('original_image')
        if original_image is None: return

        dominant_color, color_name = get_dominant_color(
            original_image, 
            mask=detection.mask,
            color_space='HSL'
        )
        detection.add_feature('dominant_color_hsl', dominant_color)
        detection.add_feature('color_name', color_name)

class HierarchicalYOLOAnalyzer(Analyzer):
    """
    在一个已检测到的物体ROI上，执行另一个YOLO模型进行二次检测。
    """
    def __init__(self):
        print("Loading hierarchical YOLO models...")
        self.secondary_models = {}
        for class_name, model_path in config.HIERARCHICAL_DETECTION_CONFIG.items():
            try:
                self.secondary_models[class_name] = YOLO(model_path)
                print(f" - Loaded '{model_path}' for class '{class_name}'")
            except Exception as e:
                print(f"Error loading secondary YOLO model {model_path}: {e}")

    def analyze(self, detection: Detection, **kwargs):
        # 检查当前检测的类别是否需要二次检测
        if detection.class_name in self.secondary_models:
            secondary_model = self.secondary_models[detection.class_name]
            roi_image = detection.roi
            
            if roi_image is None or roi_image.size == 0:
                detection.add_feature('sub_detections', [])
                return
            
            # 在ROI上运行二次检测
            sub_results = secondary_model(roi_image)
            sub_detections = []
            
            # 获取父级边界框的左上角坐标，用于坐标转换
            parent_x1, parent_y1 = detection.bbox[0], detection.bbox[1]

            for sub_res in sub_results:
                if sub_res.boxes is None: continue
                for sub_box in sub_res.boxes:
                    # 获取相对于ROI的坐标
                    x1_rel, y1_rel, x2_rel, y2_rel = map(int, sub_box.xyxy[0])
                    
                    # 转换为相对于原始大图的绝对坐标
                    x1_abs = parent_x1 + x1_rel
                    y1_abs = parent_y1 + y1_rel
                    x2_abs = parent_x1 + x2_rel
                    y2_abs = parent_y1 + y2_rel
                    
                    # 创建一个新的Detection对象来存储子检测结果
                    sub_detection_obj = Detection(
                        bbox=[x1_abs, y1_abs, x2_abs, y2_abs],
                        class_name=secondary_model.names[int(sub_box.cls)],
                        confidence=float(sub_box.conf),
                        roi=roi_image[y1_rel:y2_rel, x1_rel:x2_rel] # ROI中的ROI
                    )
                    sub_detections.append(sub_detection_obj)
            
            # 将子检测列表作为一项新特征添加到父物体中
            detection.add_feature('sub_detections', sub_detections)

class MotionAnalyzer(Analyzer):
    """判断物体是否在运动。"""
    def analyze(self, detection: Detection, **kwargs):
        motion_mask = kwargs.get('motion_mask')
        is_moving = False
        if motion_mask is not None and detection.mask is not None:
            intersection = np.bitwise_and(detection.mask, motion_mask)
            object_area = np.sum(detection.mask > 0)
            if object_area > 0 and (np.sum(intersection > 0) / object_area) > 0.2:
                is_moving = True
        detection.add_feature('is_moving', is_moving)