# backend/vision/analyzers.py (修正版)
import numpy as np
from ultralytics import YOLO
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color
from .detection import get_initial_detections
from .ocr import get_text_from_image

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
                    
                    if 'post_rules' in sub_config_node:
                        rules = sub_config_node.get('post_rules', [])
                        
                        # 按类别对子检测结果进行分组
                        grouped_sub_detections = {}
                        for sub_d in sub_detections:
                            grouped_sub_detections.setdefault(sub_d.class_name, []).append(sub_d)

                        final_sub_detections = []
                        processed_classes = set()

                        # 应用规则
                        for rule in rules:
                            rule_class = rule['class']
                            if rule_class in grouped_sub_detections:
                                processed_classes.add(rule_class)
                                class_detections = grouped_sub_detections[rule_class]
                                
                                max_detections = rule.get('max_detections', -1)
                                strategy = rule.get('strategy', 'highest_confidence')

                                if max_detections != -1 and len(class_detections) > max_detections:
                                    if strategy == 'highest_confidence':
                                        # 按置信度降序排序
                                        class_detections.sort(key=lambda d: d.confidence, reverse=True)
                                    # 只保留前 max_detections 个结果
                                    final_sub_detections.extend(class_detections[:max_detections])
                                else:
                                    final_sub_detections.extend(class_detections)
                        
                        # 将没有应用规则的类别重新加回去
                        for class_name, dets in grouped_sub_detections.items():
                            if class_name not in processed_classes:
                                final_sub_detections.extend(dets)
                        
                        # 用经过规则筛选的列表替换原始的子检测列表
                        sub_detections = final_sub_detections
                    
                    detection.add_feature('sub_detections', sub_detections)
                    
                    # --- 修正：准备下一轮递归的kwargs ---
                    if next_level_config:
                        next_kwargs = kwargs.copy()
                        next_kwargs['yolo_model'] = secondary_model
                        self._recursive_analyze(sub_detections, next_level_config, **next_kwargs)

class GridPositionAnalyzer(Analyzer):
    """
    Analyzes the position of a detection within a 3x3 grid on the image.
    Adds a 'position' attribute to the detection.
    """
    def __init__(self, image_width: int, image_height: int):
        """
        Initializes the analyzer with the dimensions of the image.
        :param image_width: The width of the image.
        :param image_height: The height of the image.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.grid_labels = [
            ["top-left", "top-center", "top-right"],
            ["middle-left", "middle-center", "middle-right"],
            ["bottom-left", "bottom-center", "bottom-right"]
        ]

    def analyze(self, detection: list[Detection]):
        """
        Determines the grid position of the detection and updates it.
        :param detection: The Detection object to analyze.
        """
        for d in detection:

            # Calculate the center of the bounding box
            box = d.bbox
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Determine the grid cell
            grid_x = int(center_x / self.image_width * 3)
            grid_y = int(center_y / self.image_height * 3)

            # Clamp values to be within the grid bounds (0-2)
            grid_x = min(grid_x, 2)
            grid_y = min(grid_y, 2)
            
            position = self.grid_labels[grid_y][grid_x]
            d.add_feature('position', position)

class OCRAnalyzer(Analyzer):
    """
    对指定的检测对象运行OCR，提取文字信息。
    """
    def analyze(self, detection: Detection, **kwargs):
        """
        如果检测对象的类别在配置文件的白名单中，则对其ROI进行OCR。
        """
        # 检查该物体的类别是否在配置文件指定的OCR名单中
        if detection.class_name in config.OCR_ENABLED_CLASSES:
            # 调用OCR函数处理物体的ROI
            recognized_text = get_text_from_image(detection.roi)
            
            # 如果识别到了文字，就将其添加到特征字典中
            if recognized_text:
                detection.add_feature('text', recognized_text)
