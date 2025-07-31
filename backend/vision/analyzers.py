# backend/vision/analyzers.py (修正版)
import numpy as np
from ultralytics import YOLO
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color
from .detection import get_initial_detections

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

class RecursiveYOLOAnalyzer(Analyzer):
    """一个可以递归执行YOLO检测的分析器。"""
    def __init__(self, initial_config, other_analyzers=None):
        print("Initializing Recursive YOLO Analyzer...")
        self.model_cache = {}
        self.initial_config = initial_config
        self.other_analyzers = other_analyzers if other_analyzers else []

    def _get_model(self, model_path):
        """动态加载并缓存YOLO模型。"""
        if model_path not in self.model_cache:
            try:
                self.model_cache[model_path] = YOLO(model_path)
                print(f" - Loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                self.model_cache[model_path] = None
        return self.model_cache[model_path]

    def analyze(self, detection: Detection, **kwargs):
        """这是递归分析的入口。"""
        if detection is None:
            image_bgr = kwargs.get('original_image')
            yolo_model = kwargs.get('yolo_model')
            initial_detections = get_initial_detections(image_bgr, yolo_model)
            
            # --- 修正：直接传递kwargs，不再额外传递image_bgr作为位置参数 ---
            self._recursive_analyze(initial_detections, self.initial_config, **kwargs)
            return initial_detections
        return []

    # --- 修正：修改函数签名，不再单独接收original_image ---
    def _recursive_analyze(self, detections: list, current_config: dict, **kwargs):
        """递归地对检测结果进行分析和再检测。"""
        original_image = kwargs.get('original_image')
        yolo_model = kwargs.get('yolo_model')

        for detection in detections:
            # --- 1. 对当前层级的每个物体应用基础分析器 ---
            analyzer_kwargs = {**kwargs, 'yolo_model': yolo_model}
            for analyzer in self.other_analyzers:
                analyzer.analyze(detection, **analyzer_kwargs)

            # --- 2. 检查是否需要进行下一层级的检测 ---
            if detection.class_name in current_config:
                sub_config_node = current_config[detection.class_name]
                model_path = sub_config_node.get('model_path')
                next_level_config = sub_config_node.get('sub_config', {})
                
                secondary_model = self._get_model(model_path)
                if secondary_model and detection.roi is not None and detection.roi.size > 0:
                    
                    sub_detections = get_initial_detections(detection.roi, secondary_model)
                    
                    parent_x1, parent_y1 = detection.bbox[0], detection.bbox[1]
                    for sub_d in sub_detections:
                        sub_d.bbox = [
                            parent_x1 + sub_d.bbox[0], parent_y1 + sub_d.bbox[1],
                            parent_x1 + sub_d.bbox[2], parent_y1 + sub_d.bbox[3]
                        ]
                    
                    detection.add_feature('sub_detections', sub_detections)
                    
                    # --- 修正：准备下一轮递归的kwargs ---
                    if next_level_config:
                        next_kwargs = kwargs.copy()
                        next_kwargs['yolo_model'] = secondary_model
                        self._recursive_analyze(sub_detections, next_level_config, **next_kwargs)