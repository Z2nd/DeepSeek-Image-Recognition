# backend/vision/pipelines.py
from .core import Detection, Analyzer
from .detection import get_initial_detections
from .drawing import draw_enhanced_annotations, draw_segmentation_masks # 导入新函数

class VisionPipeline:
    def __init__(self, yolo_model, analyzers: list[Analyzer]):
        self.yolo_model = yolo_model
        self.analyzers = analyzers

    def run(self, image_bgr, **kwargs):
        """
        执行完整的视觉分析流水线，现在包含分割掩码的绘制。
        """
        # 1. 初始检测
        detections = get_initial_detections(image_bgr, self.yolo_model)
        
        # 2. 依次运行所有分析器
        for detection in detections:
            analyzer_kwargs = {
                'original_image': image_bgr,
                'yolo_model': self.yolo_model,
                **kwargs
            }
            for analyzer in self.analyzers:
                analyzer.analyze(detection, **analyzer_kwargs)
        
        # --- 3. 核心改动：分两步进行绘图 ---
        
        # 步骤 3.1: 首先在原始图像上绘制半透明的分割掩码
        image_with_masks = draw_segmentation_masks(image_bgr, detections)
        
        # 步骤 3.2: 然后在已经带有掩码的图像上绘制边界框和标签
        # 将Detection对象列表转换为绘图函数所需的字典列表
        detections_dict_list = [
            {
                "bbox": d.bbox,
                "class": d.class_name,
                "confidence": d.confidence,
                **d.features
            } for d in detections
        ]
        final_annotated_image = draw_enhanced_annotations(image_with_masks, detections_dict_list)

        return detections, final_annotated_image