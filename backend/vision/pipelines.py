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
        # --- 新增：从kwargs中移除已明确传递的参数，增加代码稳健性 ---
        kwargs.pop('yolo_model', None)
        kwargs.pop('original_image', None)

        # 1. 启动递归分析
        all_detections = self.recursive_analyzer.analyze(
            None, 
            original_image=image_bgr, 
            yolo_model=self.yolo_model, 
            **kwargs
        )
        
        # 2. 绘制所有层级的标注
        annotated_image = draw_enhanced_annotations(image_bgr, all_detections)

        return all_detections, annotated_image