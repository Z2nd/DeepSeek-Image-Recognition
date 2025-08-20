# backend/vision/drawing.py
import cv2
import numpy as np
import random

# Create a color palette for different instance segmentation masks
# For reproducibility, we set a fixed random seed
random.seed(42)
COLOR_PALETTE = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(100)]

def draw_segmentation_masks(image: np.ndarray, detections: list):
    """
    Draw semi-transparent instance segmentation masks on the image.

    Args:
        image (np.ndarray): Original BGR image.
        detections (list): List of Detection objects.

    Returns:
        np.ndarray: Image with overlaid masks.
    """
    annotated_image = image.copy()
    img_height, img_width, _ = annotated_image.shape

    for i, detection in enumerate(detections):
        mask = detection.mask
        if mask is None:
            continue
        # Find contours from the binary mask
        # cv2.RETR_EXTERNAL finds only the outer contours of an object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select a color for this instance's contour
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        # Make the contour thickness adaptive to the image size
        thickness = max(1, int(img_width / 200))

        # Draw all found contours for this object on the image
        cv2.drawContours(annotated_image, contours, -1, color, thickness)

    # Blend the colored mask with the original image using a weighted sum
    return annotated_image

def draw_enhanced_annotations(image: np.ndarray, detections_list: list):
    """
    Draw bounding boxes and labels on the image, supporting recursive drawing and using different colors for each level.
    """
    annotated_image = image.copy()
    img_height, img_width, _ = annotated_image.shape

    # Define colors for different levels (main detection=green, second level=orange, third level=magenta, ...)
    level_colors = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

    def _draw_recursive(img, detections, level=0):
        for d in detections:
            # --- Draw detection for the current level ---
            bbox = d.bbox
            class_name = d.class_name
            conf = d.confidence
            color = d.features['color_name'] if 'color_name' in d.features else None
            
            box_color = level_colors[level % len(level_colors)]
            label_text_color = (0, 0, 0)

            box_thickness = max(1, int(img_width / (600 + level * 200)))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, box_thickness)
            
            label = f"{class_name} {conf:.2f} {color}" if color else f"{class_name} {conf:.2f}"
            font_scale = max(0.3, (bbox[2] - bbox[0]) / 350)
            font_thickness = max(1, int(font_scale * 1.5))
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y1_label = bbox[1] - 5
            if y1_label - th < 0: y1_label = bbox[1] + th + 5
            
            cv2.rectangle(img, (bbox[0], y1_label - th), (bbox[0] + tw, y1_label), box_color, -1)
            cv2.putText(img, label, (bbox[0], y1_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_text_color, font_thickness)

            # --- Recursively draw sub-detections ---
            if 'sub_detections' in d.features and d.features['sub_detections']:
                _draw_recursive(img, d.features['sub_detections'], level + 1)

    # Start drawing from the top level (level 0)
    _draw_recursive(annotated_image, detections_list, 0)
    return annotated_image