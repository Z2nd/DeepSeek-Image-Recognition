# backend/vision/core.py
from abc import ABC, abstractmethod
import numpy as np
import itertools

# 用于生成唯一ID的计数器
id_counter = itertools.count()

class Detection:
    """
    一个用于存储单个检测物体所有信息的数据类。
    """
    def __init__(self, bbox, class_name, confidence, mask=None, roi=None):
        self.id = next(id_counter) # 新增：为每个检测对象分配一个唯一ID
        self.bbox = bbox  # 边界框 [x1, y1, x2, y2]
        self.class_name = class_name  # 主要类别名称
        self.confidence = confidence  # YOLO置信度
        self.mask = mask  # 实例分割掩码 (numpy array)
        self.roi = roi    # 从原图中裁剪的感兴趣区域 (numpy array)
        self.features = {}  # 用于存储后续分析结果的字典

    def add_feature(self, name, value):
        """向该物体添加一个分析特征。"""
        self.features[name] = value

    def __repr__(self):
        return f"Detection(id={self.id}, class={self.class_name}, conf={self.confidence:.2f}, features={list(self.features.keys())})"

class Analyzer(ABC):
    """
    所有分析器的抽象基类。
    每个分析器都必须实现 analyze 方法。
    """
    @abstractmethod
    def analyze(self, detection: Detection, **kwargs):
        """
        对单个 Detection 对象进行分析，并将结果添加到其 features 字典中。
        
        Args:
            detection (Detection): 要分析的检测对象。
            **kwargs: 可能需要的额外参数 (如整张图、其他模型等)。
        """
        pass