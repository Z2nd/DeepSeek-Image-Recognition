# backend/vision/pipeline.py
from .core import Detection, Analyzer
from .detection import get_initial_detections, get_combined_detections
from .drawing import draw_enhanced_annotations, draw_segmentation_masks
from backend import config


class VisionPipeline:
    """A complete vision analysis pipeline that supports multi-model detection, recursive analysis, 
    post-processing analyzers, and visualization.
    """
    def __init__(self, recursive_analyzer: Analyzer, post_analyzers: list = None):
        """
        Initialize the VisionPipeline.

        Args:
            recursive_analyzer (Analyzer): Analyzer capable of recursive detection.
            post_analyzers (list, optional): Additional analyzers to run after recursive analysis. Defaults to None.
        """
        self.recursive_analyzer = recursive_analyzer
        self.post_analyzers = post_analyzers if post_analyzers else []
        self.model_cache = {} # Shared cache for all YOLO models

    def run(self, image_bgr, **kwargs):
        """
        Run the full vision analysis pipeline on the input image.

        The pipeline performs:
            1. Multi-model initial detections.
            2. Recursive analysis and additional post analyzers.
            3. Enhanced annotation drawing with bounding boxes and labels.
            4. Segmentation mask overlay.

        Args:
            image_bgr (np.ndarray): Input image in BGR format.
            **kwargs: Additional keyword arguments for analyzers.

        Returns:
            tuple:
                - list[Detection]: List of all Detection objects after analysis.
                - np.ndarray: Annotated image with bounding boxes and segmentation masks.
        """
        # Remove explicit parameters to avoid conflicts in kwargs
        kwargs.pop('yolo_model', None)
        kwargs.pop('original_image', None)
        
        # 1. Get combined detections from multiple models
        initial_detections = get_combined_detections(
            image_bgr,
            config.MULTI_MODEL_DETECTION_CONFIG,
            self.model_cache
        )
        
        # 2. Recursive analysis and feature extraction
        self.recursive_analyzer._recursive_analyze(
            initial_detections, 
            self.recursive_analyzer.initial_config,
            original_image=image_bgr,
            model_cache=self.model_cache,
            **kwargs
        )
            
        # 3. Run post-processing analyzers
        for post_analyzer in self.post_analyzers:
            post_analyzer.analyze(initial_detections)
            
        # 4. Draw enhanced annotations (bounding boxes, labels)
        annotated_image = draw_enhanced_annotations(image_bgr, initial_detections)
        
        # Collect all detections with masks for overlay
        all_detections_with_masks = []
        def collect_masks(detections):
            for d in detections:
                if d.mask is not None:
                    all_detections_with_masks.append(d)
                if 'sub_detections' in d.features:
                    collect_masks(d.features['sub_detections'])
        collect_masks(initial_detections)
        
        # Draw segmentation masks on the annotated image
        annotated_image = draw_segmentation_masks(annotated_image, all_detections_with_masks)

        return initial_detections, annotated_image