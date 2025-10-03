# backend/vision/analyzers.py
import numpy as np
import cv2
from ultralytics import YOLO
from backend import config
from .core import Analyzer, Detection
from .features import get_dominant_color
from .detection import get_initial_detections
from .ocr import get_text_from_image
from .color_clustering import dominant_colors

class ColorAnalyzer(Analyzer):
    """Analyzes the dominant color of a detected object."""
    def analyze(self, detection: Detection, **kwargs):
        """
        Extracts the dominant colors from the detected object's mask region
        and adds them as features to the detection.

        Args:
            detection (Detection): The object to analyze.
            **kwargs: Additional arguments, expects 'original_image'.
        """
        original_image = kwargs.get('original_image')
        if original_image is None: return

        # Convert the image from BGR to RGB format for color processing
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # Extract dominant colors from the object region
        palette = dominant_colors(original_image, mask=detection.mask)

        # Add the top two color names as features
        detection.add_feature('color_name', f'{palette[0]["name"] if palette else None} + {palette[1]["name"] if len(palette) > 1 else None}')

class RecursiveYOLOAnalyzer(Analyzer):
    """Performs recursive YOLO-based detection and analysis."""
    def __init__(self, initial_config, other_analyzers=None):
        """
        Initializes the analyzer with configuration and optional additional analyzers.

        Args:
            initial_config (dict): The configuration for recursive detection.
            other_analyzers (list[Analyzer], optional): Additional analyzers to apply at each detection level.
        """
        print("Initializing Recursive YOLO Analyzer...")
        self.model_cache = {}
        self.initial_config = initial_config
        self.other_analyzers = other_analyzers if other_analyzers else []

    def _get_model(self, model_path):
        """
        Loads and caches a YOLO model given its file path.

        Args:
            model_path (str): Path to the YOLO model file.

        Returns:
            YOLO: Loaded YOLO model or None if loading failed.
        """
        if model_path not in self.model_cache:
            try:
                self.model_cache[model_path] = YOLO(model_path)
                print(f" - Loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                self.model_cache[model_path] = None
        return self.model_cache[model_path]

    def analyze(self, detection: Detection, **kwargs):
        """
        Entry point for recursive analysis. If detection is None, performs
        initial detection on the provided image.

        Args:
            detection (Detection or None): Detection to analyze, or None for initial detection.
            **kwargs: Additional arguments, expects 'original_image' and 'yolo_model'.

        Returns:
            list[Detection]: List of detections after analysis.
        """
        if detection is None:
            image_bgr = kwargs.get('original_image')
            yolo_model = kwargs.get('yolo_model')
            initial_detections = get_initial_detections(image_bgr, yolo_model)
            
            # --- 修正：直接传递kwargs，不再额外传递image_bgr作为位置参数 ---
            self._recursive_analyze(initial_detections, self.initial_config, **kwargs)
            return initial_detections
        return []

    def _recursive_analyze(self, detections: list, current_config: dict, **kwargs):
        """
        Recursively analyzes detections and applies secondary YOLO models
        based on configuration.

        Args:
            detections (list[Detection]): Detections to analyze at current level.
            current_config (dict): Recursive detection configuration.
            **kwargs: Additional arguments, includes 'original_image' and 'yolo_model'.
        """
        original_image = kwargs.get('original_image')
        yolo_model = kwargs.get('yolo_model')

        for detection in detections:
            # Apply additional analyzers to each detection
            analyzer_kwargs = {**kwargs, 'yolo_model': yolo_model}
            for analyzer in self.other_analyzers:
                analyzer.analyze(detection, **analyzer_kwargs)

            # Check if the current detection triggers secondary detection
            if detection.class_name in current_config:
                sub_config_node = current_config[detection.class_name]
                model_path = sub_config_node.get('model_path')
                next_level_config = sub_config_node.get('sub_config', {})
                
                secondary_model = self._get_model(model_path)
                if secondary_model and detection.roi is not None and detection.roi.size > 0:

                    # Perform secondary detection within the ROI
                    sub_detections = get_initial_detections(detection.roi, secondary_model)
                    
                    # Adjust bounding boxes to global image coordinates
                    parent_x1, parent_y1 = detection.bbox[0], detection.bbox[1]
                    for sub_d in sub_detections:
                        sub_d.bbox = [
                            parent_x1 + sub_d.bbox[0], parent_y1 + sub_d.bbox[1],
                            parent_x1 + sub_d.bbox[2], parent_y1 + sub_d.bbox[3]
                        ]
                    
                    # Apply post-processing rules if defined
                    if 'post_rules' in sub_config_node:
                        rules = sub_config_node.get('post_rules', [])
                        
                        # Handle wildcard rules
                        wildcard_rule = next((rule for rule in rules if rule.get('class') == '*'), None)

                        if wildcard_rule:
                            max_detections = wildcard_rule.get('max_detections', -1)
                            strategy = wildcard_rule.get('strategy', 'highest_confidence')

                            if max_detections != -1 and len(sub_detections) > max_detections:
                                if strategy == 'highest_confidence':
                                    sub_detections.sort(key=lambda d: d.confidence, reverse=True)
                                sub_detections = sub_detections[:max_detections]
                        
                        else:
                            # Apply rules per class
                            grouped_sub_detections = {}
                            for sub_d in sub_detections:
                                grouped_sub_detections.setdefault(sub_d.class_name, []).append(sub_d)

                            final_sub_detections = []
                            processed_classes = set()

                            for rule in rules:
                                rule_class = rule['class']
                                if rule_class in grouped_sub_detections:
                                    processed_classes.add(rule_class)
                                    class_detections = grouped_sub_detections[rule_class]
                                    
                                    max_detections = rule.get('max_detections', -1)
                                    strategy = rule.get('strategy', 'highest_confidence')

                                    if max_detections != -1 and len(class_detections) > max_detections:
                                        if strategy == 'highest_confidence':
                                            class_detections.sort(key=lambda d: d.confidence, reverse=True)
                                        final_sub_detections.extend(class_detections[:max_detections])
                                    else:
                                        final_sub_detections.extend(class_detections)
                            
                            # Re-add detections without rules
                            for class_name, dets in grouped_sub_detections.items():
                                if class_name not in processed_classes:
                                    final_sub_detections.extend(dets)
                            
                            sub_detections = final_sub_detections
                    detection.add_feature('sub_detections', sub_detections)
                    
                    # Prepare kwargs for next recursive level
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
        Computes the grid cell for each detection and updates its features.

        Args:
            detections (list[Detection]): List of detections to analyze.
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
    Performs OCR on specific detected objects to extract text information.
    """
    def analyze(self, detection: Detection, **kwargs):
        """
        Runs OCR if the detection's class is in the OCR-enabled whitelist.

        Args:
            detection (Detection): The object to analyze.
            **kwargs: Additional arguments (unused here).
        """
        if detection.class_name in config.OCR_ENABLED_CLASSES:
            recognized_text = get_text_from_image(detection.roi)
            
            if recognized_text:
                detection.add_feature('text', recognized_text)
