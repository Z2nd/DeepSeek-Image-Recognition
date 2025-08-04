from .core import Detection
from .analyzers import RecursiveYOLOAnalyzer
from .drawing import draw_enhanced_annotations, draw_segmentation_masks

class VisionPipeline:
    def __init__(self, yolo_model, recursive_analyzer: RecursiveYOLOAnalyzer):
        self.yolo_model = yolo_model
        self.recursive_analyzer = recursive_analyzer

    def run(self, image_bgr, **kwargs):
        """
        执行视觉分析流水线，核心是调用递归分析器。
        """
        # --- Remove explicitly passed parameters from kwargs to increase code robustness ---
        kwargs.pop('yolo_model', None)
        kwargs.pop('original_image', None)

        # 1. Initiate recursive analysis
        all_detections = self.recursive_analyzer.analyze(
            None, 
            original_image=image_bgr, 
            yolo_model=self.yolo_model, 
            **kwargs
        )
        
        # 2. Drawing all layers of annotations
        annotated_image = draw_enhanced_annotations(image_bgr, all_detections)

        return all_detections, annotated_image