# backend/vision/drawing.py
import cv2
import numpy as np
import random

# 创建一个颜色列表用于不同的实例分割掩码
# 为了可复现性，我们固定随机种子
random.seed(42)
COLOR_PALETTE = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(100)]

def draw_segmentation_masks(image: np.ndarray, detections: list, alpha: float = 0.4):
    """
    在图像上绘制半透明的实例分割掩码。

    Args:
        image (np.ndarray): 原始BGR图像。
        detections (list): Detection对象的列表。
        alpha (float): 叠加掩码的透明度。

    Returns:
        np.ndarray: 叠加了掩码的图像。
    """
    overlay = image.copy()
    for i, detection in enumerate(detections):
        mask = detection.mask
        if mask is None:
            continue

        # 为每个实例选择一个颜色
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        
        # 将掩码应用到图像上
        # mask > 0 的部分会被着色
        overlay[mask > 0] = color

    # 将带有颜色的遮罩与原始图像按权重混合
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_enhanced_annotations(image: np.ndarray, detections_list: list):
    """
    在图像上绘制边界框和标签，支持递归绘制并用不同颜色区分层级。
    """
    annotated_image = image.copy()
    img_height, img_width, _ = annotated_image.shape
    
    # 定义不同层级的颜色 (主检测=绿色, 二级=橙色, 三级=洋红, ...)
    level_colors = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

    def _draw_recursive(img, detections, level=0):
        for d in detections:
            # --- 绘制当前层级的检测 ---
            bbox = d.bbox
            class_name = d.class_name
            conf = d.confidence
            
            box_color = level_colors[level % len(level_colors)]
            label_text_color = (0, 0, 0)

            box_thickness = max(1, int(img_width / (600 + level * 200)))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, box_thickness)
            
            label = f"{class_name} {conf:.2f}"
            font_scale = max(0.3, (bbox[2] - bbox[0]) / 350)
            font_thickness = max(1, int(font_scale * 1.5))
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y1_label = bbox[1] - 5
            if y1_label - th < 0: y1_label = bbox[1] + th + 5
            
            cv2.rectangle(img, (bbox[0], y1_label - th), (bbox[0] + tw, y1_label), box_color, -1)
            cv2.putText(img, label, (bbox[0], y1_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_text_color, font_thickness)

            # --- 递归调用绘制子检测 ---
            if 'sub_detections' in d.features and d.features['sub_detections']:
                _draw_recursive(img, d.features['sub_detections'], level + 1)

    # 从顶层（level 0）开始绘制
    _draw_recursive(annotated_image, detections_list, 0)
    return annotated_image