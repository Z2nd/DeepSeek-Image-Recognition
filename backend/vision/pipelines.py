from .core import Detection, Analyzer
from .detection import get_initial_detections, get_combined_detections
from .drawing import draw_enhanced_annotations, draw_segmentation_masks
from backend import config


class VisionPipeline:
    def __init__(self, recursive_analyzer: Analyzer, post_analyzers: list = None):
        self.recursive_analyzer = recursive_analyzer
        self.post_analyzers = post_analyzers if post_analyzers else [] # 新增
        self.model_cache = {} # <-- 新增：为所有模型提供一个共享缓存

    def run(self, image_bgr, **kwargs):
        """
        执行视觉分析流水线，并增加了后处理步骤。
        """
        # --- Remove explicitly passed parameters from kwargs to increase code robustness ---
        kwargs.pop('yolo_model', None)
        kwargs.pop('original_image', None)
        
        # --- 1. 从多个模型获取组合后的初始检测结果 ---
        initial_detections = get_combined_detections(
            image_bgr,
            config.MULTI_MODEL_DETECTION_CONFIG,
            self.model_cache
        )
        
        # --- 2. 对合并后的结果列表进行递归分析和其他分析 ---
        # 我们需要调用 recursive_analyzer 的内部方法来处理已经存在的检测列表
        self.recursive_analyzer._recursive_analyze(
            initial_detections, 
            self.recursive_analyzer.initial_config,
            original_image=image_bgr,
            model_cache=self.model_cache, # 将缓存传递下去
            **kwargs
        )
            
        # --- 3. 运行所有的后处理分析器 ---
        for post_analyzer in self.post_analyzers:
            # 注意：GridPositionAnalyzer可能需要修改来处理列表
            post_analyzer.analyze(initial_detections)
            
        # --- 4. 绘制所有层级的标注 ---
        annotated_image = draw_enhanced_annotations(image_bgr, initial_detections)
        
        # 找到所有包含掩码的检测对象并绘制
        all_detections_with_masks = []
        def collect_masks(detections):
            for d in detections:
                if d.mask is not None:
                    all_detections_with_masks.append(d)
                if 'sub_detections' in d.features:
                    collect_masks(d.features['sub_detections'])
        collect_masks(initial_detections)
        
        annotated_image = draw_segmentation_masks(annotated_image, all_detections_with_masks)

        return initial_detections, annotated_image