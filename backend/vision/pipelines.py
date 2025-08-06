from .core import Detection, Analyzer
from .detection import get_initial_detections
from .drawing import draw_enhanced_annotations


class VisionPipeline:
    def __init__(self, yolo_model, recursive_analyzer: Analyzer, post_analyzers: list = None):
        self.yolo_model = yolo_model
        self.recursive_analyzer = recursive_analyzer
        self.post_analyzers = post_analyzers if post_analyzers else [] # 新增

    def run(self, image_bgr, **kwargs):
        """
        执行视觉分析流水线，并增加了后处理步骤。
        """
        # --- Remove explicitly passed parameters from kwargs to increase code robustness ---
        kwargs.pop('yolo_model', None)
        kwargs.pop('original_image', None)
        
        # 1. 启动递归分析，获取所有层级的检测结果
        all_detections = self.recursive_analyzer.analyze(
            None, 
            original_image=image_bgr, 
            yolo_model=self.yolo_model, 
            **kwargs
        )
        
        # --- 2. 新增：运行所有的后处理分析器 ---
        for post_analyzer in self.post_analyzers:
            post_analyzer.analyze_all(all_detections)
            
        # 3. 绘制所有层级的标注
        annotated_image = draw_enhanced_annotations(image_bgr, all_detections)

        return all_detections, annotated_image