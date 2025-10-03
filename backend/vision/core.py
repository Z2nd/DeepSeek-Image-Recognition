# backend/vision/core.py
from abc import ABC, abstractmethod
import numpy as np
import itertools

# Counter for generating unique IDs for each detection
id_counter = itertools.count()

class Detection:
    """
    Data class representing a single detected object with all associated information.

    Attributes:
        id (int): Unique ID assigned to this detection.
        bbox (list[float]): Bounding box coordinates [x1, y1, x2, y2].
        class_name (str): Primary class label of the object.
        confidence (float): Confidence score from YOLO detection.
        mask (np.ndarray or None): Optional instance segmentation mask.
        roi (np.ndarray or None): Optional region of interest cropped from the original image.
        features (dict): Dictionary to store additional analysis results.
    """
    def __init__(self, bbox, class_name, confidence, mask=None, roi=None):
        self.id = next(id_counter) # Assign a unique ID to each detection
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.mask = mask
        self.roi = roi
        self.features = {}

    def add_feature(self, name, value):
        """
        Adds an analysis feature to this detection.

        Args:
            name (str): Name of the feature.
            value: Value of the feature.
        """
        self.features[name] = value

    def __repr__(self):
        return f"Detection(id={self.id}, class={self.class_name}, conf={self.confidence:.2f}, features={list(self.features.keys())})"

class Analyzer(ABC):
    """
    Abstract base class for all analyzers.
    Each analyzer must implement the `analyze` method.
    """
    @abstractmethod
    def analyze(self, detection: Detection, **kwargs):
        """
        Analyze a single Detection object and add results to its features dictionary.

        Args:
            detection (Detection): The detection object to analyze.
            **kwargs: Additional optional parameters (e.g., original image, other models).
        """
        pass