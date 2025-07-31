# backend/vision/analyzers.py
import numpy as np
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color
from .classification import secondary_classification

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
            color_space='HSV'
        )
        detection.add_feature('dominant_color_hsv', dominant_color)
        detection.add_feature('color_name', color_name)

class SecondaryClassifierAnalyzer(Analyzer):
    """执行二次分类以获取子类别。"""
    def __init__(self, secondary_model):
        self.secondary_model = secondary_model

    def analyze(self, detection: Detection, **kwargs):
        group = detection.features.get('group', 'other')
        sub_class, sub_conf = 'unknown', 0.0

        if self.secondary_model and group in ['animal', 'vehicle']:
            class_map = config.ANIMAL_SUBCLASSES if group == 'animal' else config.VEHICLE_SUBCLASSES
            class_names = class_map.get(detection.class_name, [])
            if class_names:
                sub_class, sub_conf = secondary_classification(detection.roi, self.secondary_model, class_names)
        
        detection.add_feature('sub_class', sub_class)
        detection.add_feature('sub_confidence', sub_conf)

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