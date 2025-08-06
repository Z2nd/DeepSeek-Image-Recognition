# backend/vision/analyzers.py (修正版)
import numpy as np
from ultralytics import YOLO
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color
from .detection import get_initial_detections

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
        # detection.add_feature('dominant_color_hsv', dominant_color)
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

class SpatialAnalyzer:
    """
    一个后处理分析器，用于分析所有检测对象之间的空间关系。
    注意：它的接口与标准的Analyzer不同，因为它需要处理整个列表。
    """
    def __init__(self, tolerance=1.5):
        # 容差因子，用于判断主方向。值越大，对角线方向越容易被判定为纯粹的上下或左右。
        self.tolerance = tolerance

    def _get_centroid(self, bbox):
        """计算边界框的中心点。"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _is_inside(self, bbox1, bbox2):
        """判断bbox1是否在bbox2内部。"""
        return bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and \
               bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]

    def analyze_all(self, detections: list[Detection]):
        """
        分析列表中所有检测对象两两之间的空间关系。
        """
        # 为每个detection对象初始化空间关系列表
        for d in detections:
            d.add_feature('spatial_relationships', [])

        # 使用双层循环比较每一对不同的物体
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i == j:
                    continue

                d1 = detections[i]
                d2 = detections[j]

                # 1. 检查包含关系
                if self._is_inside(d1.bbox, d2.bbox):
                    d1.features['spatial_relationships'].append(f"inside '{d2.class_name}' (id:{d2.id})")
                    continue # 如果在内部，则不再判断方向关系

                # 2. 检查方向关系
                cx1, cy1 = self._get_centroid(d1.bbox)
                cx2, cy2 = self._get_centroid(d2.bbox)
                
                dx = cx1 - cx2
                dy = cy1 - cy2

                # 根据容差判断主导方向
                if abs(dx) > abs(dy) * self.tolerance:
                    if dx < 0:
                        d1.features['spatial_relationships'].append(f"left_of '{d2.class_name}' (id:{d2.id})")
                    else:
                        d1.features['spatial_relationships'].append(f"right_of '{d2.class_name}' (id:{d2.id})")
                elif abs(dy) > abs(dx) * self.tolerance:
                    if dy < 0:
                        d1.features['spatial_relationships'].append(f"above '{d2.class_name}' (id:{d2.id})")
                    else:
                        d1.features['spatial_relationships'].append(f"below '{d2.class_name}' (id:{d2.id})")
